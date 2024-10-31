from copy import deepcopy
import hashlib
import math

import os
from urllib.parse import urlparse
import cv2
import torch
from torch.hub import download_url_to_file, get_dir
import numpy as np

from seemore.module import SeemoRe


def md5sum(filename):
    md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(128 * md5.block_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def get_cache_path_by_url(url):
    parts = urlparse(url)
    model_dir = os.path.join(get_dir(), "checkpoints")
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(model_dir, filename)
    return cached_file


def download_model(url, model_md5: str = None):
    if os.path.exists(url):
        cached_file = url
    else:
        cached_file = get_cache_path_by_url(url)
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}')
        hash_prefix = None
        download_url_to_file(url, cached_file, hash_prefix, progress=True)
        if model_md5:
            _md5 = md5sum(cached_file)
            if model_md5 != _md5:
                os.remove(cached_file)
                raise ValueError(
                    f"Model md5: {_md5}, expected md5: {model_md5}, wrong model deleted"
                )

    return cached_file


seemore_t_base_cfg = {
    "in_chans": 3,
    "num_experts": 3,
    "img_range": 1.0,
    "num_layers": 6,
    "embedding_dim": 36,
    "use_shuffle": True,
    "lr_space": "exp",
    "topk": 1,
    "recursive": 2,
    "global_kernel_size": 11,
}

seemore_b_base_cfg = {
    "in_chans": 3,
    "num_experts": 3,
    "img_range": 1.0,
    "num_layers": 8,
    "embedding_dim": 48,
    "use_shuffle": True,
    "lr_space": "exp",
    "topk": 1,
    "recursive": 2,
    "global_kernel_size": 11,
}


seemore_model_cfgs = {
    "seemore_b_x2": {
        "scale": 2,
        "url": "https://github.com/Sanster/seemore/releases/download/Models/SeemoRe_B_X2.pth",
        **seemore_b_base_cfg,
    },
    "seemore_b_x3": {
        "scale": 3,
        "url": "https://github.com/Sanster/seemore/releases/download/Models/SeemoRe_B_X3.pth",
        **seemore_b_base_cfg,
    },
    "seemore_b_x4": {
        "scale": 4,
        "url": "https://github.com/Sanster/seemore/releases/download/Models/SeemoRe_B_X4.pth",
        **seemore_b_base_cfg,
    },
    "seemore_t_x2": {
        "scale": 2,
        "url": "https://github.com/Sanster/seemore/releases/download/Models/SeemoRe_T_X2.pth",
        **seemore_t_base_cfg,
    },
    "seemore_t_x3": {
        "scale": 3,
        "url": "https://github.com/Sanster/seemore/releases/download/Models/SeemoRe_T_X3.pth",
        **seemore_t_base_cfg,
    },
    "seemore_t_x4": {
        "scale": 4,
        "url": "https://github.com/Sanster/seemore/releases/download/Models/SeemoRe_T_X4.pth",
        **seemore_t_base_cfg,
    },
}


class SeemoReUpscaler:
    IMAGE_MODE_GRAY = 1
    IMAGE_MODE_BGRA = 2
    IMAGE_MODE_BGR = 3

    def __init__(self, model_name: str, device: str = "cpu"):
        if model_name not in seemore_model_cfgs:
            raise ValueError(
                f"Model {model_name} not found, available models: {list(seemore_model_cfgs.keys())}"
            )
        cfg = deepcopy(seemore_model_cfgs[model_name])
        cfg.pop("url")
        self.model = SeemoRe(**cfg)
        ckpt_path = download_model(seemore_model_cfgs[model_name]["url"])
        state_dict = torch.load(ckpt_path, map_location="cpu")["params"]
        self.model.load_state_dict(state_dict, strict=True)
        self.model = self.model.to(device)
        self.device = device

    @torch.inference_mode()
    def __call__(
        self, np_img: np.ndarray, tile_size: int = 0, scale: float = 0
    ) -> np.ndarray:
        original_h, original_w = np_img.shape[:2]
        if np_img.ndim == 2 or (np_img.ndim == 3 and np_img.shape[2] == 1):
            image_mode = self.IMAGE_MODE_GRAY
            rgb_np_img = cv2.cvtColor(np_img, cv2.COLOR_GRAY2RGB)
        elif np_img.ndim == 3 and np_img.shape[2] == 4:
            image_mode = self.IMAGE_MODE_BGRA
            alpha = np_img[:, :, 3]
            np_img = np_img[:, :, 0:3]
            rgb_np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
        else:
            image_mode = self.IMAGE_MODE_BGR
            rgb_np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)

        y = torch.tensor(rgb_np_img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        y = y / 255.0
        if tile_size > 0:
            x_hat = self.tile_inference(y, tile_size)
        else:
            x_hat = self.model(y)
        restored_img = (
            x_hat.squeeze().permute(1, 2, 0).clamp_(0, 1).cpu().detach().numpy()
        )
        restored_img = np.clip(restored_img, 0.0, 1.0)
        restored_img = (restored_img * 255.0).round().astype(np.uint8)

        if image_mode == self.IMAGE_MODE_GRAY:
            restored_img = cv2.cvtColor(restored_img, cv2.COLOR_RGB2GRAY)
        elif image_mode == self.IMAGE_MODE_BGRA:
            # Handle alpha channel
            h, w = alpha.shape[0:2]
            upscaled_alpha = cv2.resize(
                alpha,
                (w * self.model.scale, h * self.model.scale),
                interpolation=cv2.INTER_LINEAR,
            )
            restored_img = cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGRA)
            restored_img[:, :, 3] = upscaled_alpha
        else:  # BGR mode
            restored_img = cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR)

        if scale > 0 and scale != self.model.scale:
            restored_img = cv2.resize(
                restored_img,
                (original_w * scale, original_h * scale),
                interpolation=cv2.INTER_LANCZOS4,
            )

        return restored_img

    @torch.inference_mode()
    def tile_inference(self, y: torch.Tensor, tile_size: int) -> torch.Tensor:
        # https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/utils.py#L117
        batch, channel, height, width = y.shape
        output_height = height * self.model.scale
        output_width = width * self.model.scale
        output_shape = (batch, channel, output_height, output_width)

        # Initialize output tensor
        output = y.new_zeros(output_shape)

        # Calculate number of tiles
        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)

        # Padding size
        tile_pad = 12

        # Process each tile
        for y_idx in range(tiles_y):
            for x_idx in range(tiles_x):
                # Calculate tile boundaries
                x_start = x_idx * tile_size
                y_start = y_idx * tile_size
                x_end = min(x_start + tile_size, width)
                y_end = min(y_start + tile_size, height)

                # Add padding
                x_start_pad = max(x_start - tile_pad, 0)
                x_end_pad = min(x_end + tile_pad, width)
                y_start_pad = max(y_start - tile_pad, 0)
                y_end_pad = min(y_end + tile_pad, height)

                # Extract tile with padding
                tile = y[:, :, y_start_pad:y_end_pad, x_start_pad:x_end_pad]

                # Process tile
                output_tile = self.model(tile)

                # Calculate output coordinates
                out_x_start = x_start * self.model.scale
                out_x_end = x_end * self.model.scale
                out_y_start = y_start * self.model.scale
                out_y_end = y_end * self.model.scale

                # Calculate valid output region (removing padding)
                out_x_start_valid = (x_start - x_start_pad) * self.model.scale
                out_x_end_valid = (
                    out_x_start_valid + (x_end - x_start) * self.model.scale
                )
                out_y_start_valid = (y_start - y_start_pad) * self.model.scale
                out_y_end_valid = (
                    out_y_start_valid + (y_end - y_start) * self.model.scale
                )

                # Place valid region into output
                output[:, :, out_y_start:out_y_end, out_x_start:out_x_end] = (
                    output_tile[
                        :,
                        :,
                        out_y_start_valid:out_y_end_valid,
                        out_x_start_valid:out_x_end_valid,
                    ]
                )

        return output
