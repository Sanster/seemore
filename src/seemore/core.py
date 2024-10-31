import math
from typing import List

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


######################
# Meta Architecture
######################
class SeemoRe(nn.Module):
    def __init__(
        self,
        scale: int = 4,
        in_chans: int = 3,
        num_experts: int = 6,
        num_layers: int = 6,
        embedding_dim: int = 64,
        img_range: float = 1.0,
        use_shuffle: bool = False,
        global_kernel_size: int = 11,
        recursive: int = 2,
        lr_space: int = 1,
        topk: int = 2,
    ):
        super().__init__()
        self.scale = scale
        self.num_in_channels = in_chans
        self.num_out_channels = in_chans
        self.img_range = img_range

        rgb_mean = (0.4488, 0.4371, 0.4040)
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        # -- SHALLOW FEATURES --
        self.conv_1 = nn.Conv2d(
            self.num_in_channels, embedding_dim, kernel_size=3, padding=1
        )

        # -- DEEP FEATURES --
        self.body = nn.ModuleList(
            [
                ResGroup(
                    in_ch=embedding_dim,
                    num_experts=num_experts,
                    use_shuffle=use_shuffle,
                    topk=topk,
                    lr_space=lr_space,
                    recursive=recursive,
                    global_kernel_size=global_kernel_size,
                )
                for i in range(num_layers)
            ]
        )

        # -- UPSCALE --
        self.norm = LayerNorm(embedding_dim, data_format="channels_first")
        self.conv_2 = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, padding=1)
        self.upsampler = nn.Sequential(
            nn.Conv2d(
                embedding_dim,
                (scale**2) * self.num_out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.PixelShuffle(scale),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        # -- SHALLOW FEATURES --
        x = self.conv_1(x)
        res = x

        # -- DEEP FEATURES --
        for idx, layer in enumerate(self.body):
            x = layer(x)

        x = self.norm(x)

        # -- HR IMAGE RECONSTRUCTION --
        x = self.conv_2(x) + res
        x = self.upsampler(x)

        x = x / self.img_range + self.mean
        return x


#############################
# Components
#############################
class ResGroup(nn.Module):
    def __init__(
        self,
        in_ch: int,
        num_experts: int,
        global_kernel_size: int = 11,
        lr_space: int = 1,
        topk: int = 2,
        recursive: int = 2,
        use_shuffle: bool = False,
    ):
        super().__init__()

        self.local_block = RME(
            in_ch=in_ch,
            num_experts=num_experts,
            use_shuffle=use_shuffle,
            lr_space=lr_space,
            topk=topk,
            recursive=recursive,
        )
        self.global_block = SME(in_ch=in_ch, kernel_size=global_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.local_block(x)
        x = self.global_block(x)
        return x


#############################
# Global Block
#############################
class SME(nn.Module):
    def __init__(self, in_ch: int, kernel_size: int = 11):
        super().__init__()

        self.norm_1 = LayerNorm(in_ch, data_format="channels_first")
        self.block = StripedConvFormer(in_ch=in_ch, kernel_size=kernel_size)

        self.norm_2 = LayerNorm(in_ch, data_format="channels_first")
        self.ffn = GatedFFN(in_ch, mlp_ratio=2, kernel_size=3, act_layer=nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(self.norm_1(x)) + x
        x = self.ffn(self.norm_2(x)) + x
        return x


class StripedConvFormer(nn.Module):
    def __init__(self, in_ch: int, kernel_size: int):
        super().__init__()
        self.in_ch = in_ch
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.proj = nn.Conv2d(in_ch, in_ch, kernel_size=1, padding=0)
        self.to_qv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch * 2, kernel_size=1, padding=0),
            nn.GELU(),
        )

        self.attn = StripedConv2d(in_ch, kernel_size=kernel_size, depthwise=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, v = self.to_qv(x).chunk(2, dim=1)
        q = self.attn(q)
        x = self.proj(q * v)
        return x


#############################
# Local Blocks
#############################
class RME(nn.Module):
    def __init__(
        self,
        in_ch: int,
        num_experts: int,
        topk: int,
        lr_space: int = 1,
        recursive: int = 2,
        use_shuffle: bool = False,
    ):
        super().__init__()

        self.norm_1 = LayerNorm(in_ch, data_format="channels_first")
        self.block = MoEBlock(
            in_ch=in_ch,
            num_experts=num_experts,
            topk=topk,
            use_shuffle=use_shuffle,
            recursive=recursive,
            lr_space=lr_space,
        )

        self.norm_2 = LayerNorm(in_ch, data_format="channels_first")
        self.ffn = GatedFFN(in_ch, mlp_ratio=2, kernel_size=3, act_layer=nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(self.norm_1(x)) + x
        x = self.ffn(self.norm_2(x)) + x
        return x


#################
# MoE Layer
#################
class MoEBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        num_experts: int,
        topk: int,
        use_shuffle: bool = False,
        lr_space: str = "linear",
        recursive: int = 2,
    ):
        super().__init__()
        self.use_shuffle = use_shuffle
        self.recursive = recursive

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_ch, 2 * in_ch, kernel_size=1, padding=0),
        )

        self.agg_conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=4, stride=4, groups=in_ch), nn.GELU()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch),
            nn.Conv2d(in_ch, in_ch, kernel_size=1, padding=0),
        )

        self.conv_2 = nn.Sequential(
            StripedConv2d(in_ch, kernel_size=3, depthwise=True), nn.GELU()
        )

        if lr_space == "linear":
            grow_func = lambda i: i + 2
        elif lr_space == "exp":
            grow_func = lambda i: 2 ** (i + 1)
        elif lr_space == "double":
            grow_func = lambda i: 2 * i + 2
        else:
            raise NotImplementedError(f"lr_space {lr_space} not implemented")

        self.moe_layer = MoELayer(
            experts=[
                Expert(in_ch=in_ch, low_dim=grow_func(i)) for i in range(num_experts)
            ],  # add here multiple of 2 as low_dim
            gate=Router(in_ch=in_ch, num_experts=num_experts),
            num_expert=topk,
        )

        self.proj = nn.Conv2d(in_ch, in_ch, kernel_size=1, padding=0)

    def calibrate(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        res = x

        for _ in range(self.recursive):
            x = self.agg_conv(x)
        x = self.conv(x)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        return res + x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x)

        if self.use_shuffle:
            x = channel_shuffle(x, groups=2)
        x, k = torch.chunk(x, chunks=2, dim=1)

        x = self.conv_2(x)
        k = self.calibrate(k)

        x = self.moe_layer(x, k)
        x = self.proj(x)
        return x


class MoELayer(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module, num_expert: int = 1):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.num_expert = num_expert

    def forward(self, inputs: torch.Tensor, k: torch.Tensor):
        out = self.gate(inputs)
        weights = F.softmax(out, dim=1, dtype=torch.float).to(inputs.dtype)
        topk_weights, topk_experts = torch.topk(weights, self.num_expert)
        out = inputs.clone()

        if self.training:
            exp_weights = torch.zeros_like(weights)
            exp_weights.scatter_(1, topk_experts, weights.gather(1, topk_experts))
            for i, expert in enumerate(self.experts):
                out += expert(inputs, k) * exp_weights[:, i : i + 1, None, None]
        else:
            selected_experts = [self.experts[i] for i in topk_experts.squeeze(dim=0)]
            for i, expert in enumerate(selected_experts):
                out += expert(inputs, k) * topk_weights[:, i : i + 1, None, None]

        return out


class Expert(nn.Module):
    def __init__(
        self,
        in_ch: int,
        low_dim: int,
    ):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_ch, low_dim, kernel_size=1, padding=0)
        self.conv_2 = nn.Conv2d(in_ch, low_dim, kernel_size=1, padding=0)
        self.conv_3 = nn.Conv2d(low_dim, in_ch, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x)
        x = self.conv_2(k) * x  # here no more sigmoid
        x = self.conv_3(x)
        return x


class Squeeze(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(-1).squeeze(-1)


class Router(nn.Module):
    def __init__(self, in_ch: int, num_experts: int):
        super().__init__()

        self.body = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Squeeze(),
            nn.Linear(in_ch, num_experts, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


#################
# Utilities
#################
class StripedConv2d(nn.Module):
    def __init__(self, in_ch: int, kernel_size: int, depthwise: bool = False):
        super().__init__()
        self.in_ch = in_ch
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_ch,
                in_ch,
                kernel_size=(1, self.kernel_size),
                padding=(0, self.padding),
                groups=in_ch if depthwise else 1,
            ),
            nn.Conv2d(
                in_ch,
                in_ch,
                kernel_size=(self.kernel_size, 1),
                padding=(self.padding, 0),
                groups=in_ch if depthwise else 1,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def channel_shuffle(x, groups=2):
    bat_size, channels, w, h = x.shape
    group_c = channels // groups
    x = x.view(bat_size, groups, group_c, w, h)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(bat_size, -1, w, h)
    return x


class GatedFFN(nn.Module):
    def __init__(
        self,
        in_ch,
        mlp_ratio,
        kernel_size,
        act_layer,
    ):
        super().__init__()
        mlp_ch = in_ch * mlp_ratio

        self.fn_1 = nn.Sequential(
            nn.Conv2d(in_ch, mlp_ch, kernel_size=1, padding=0),
            act_layer,
        )
        self.fn_2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, padding=0),
            act_layer,
        )

        self.gate = nn.Conv2d(
            mlp_ch // 2,
            mlp_ch // 2,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=mlp_ch // 2,
        )

    def feat_decompose(self, x):
        s = x - self.gate(x)
        x = x + self.sigma * s
        return x

    def forward(self, x: torch.Tensor):
        x = self.fn_1(x)
        x, gate = torch.chunk(x, 2, dim=1)

        gate = self.gate(gate)
        x = x * gate

        x = self.fn_2(x)
        return x


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


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

seemore_model_cfgs = {}
for model_type in ["b", "t"]:
    base_cfg = seemore_b_base_cfg if model_type == "b" else seemore_t_base_cfg
    for scale in [2, 3, 4]:
        model_name = f"seemore_{model_type}_x{scale}"
        seemore_model_cfgs[model_name] = {"scale": scale, **base_cfg}


class SeemoReUpscaler:
    IMAGE_MODE_GRAY = 1
    IMAGE_MODE_BGRA = 2
    IMAGE_MODE_BGR = 3

    def __init__(self, model_name: str, device: str = "cpu"):
        if model_name not in seemore_model_cfgs:
            raise ValueError(
                f"Model {model_name} not found, available models: {list(seemore_model_cfgs.keys())}"
            )
        self.model = SeemoRe(**seemore_model_cfgs[model_name])
        ckpt_path = ""
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
