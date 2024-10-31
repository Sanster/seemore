"""测试模块"""

import os
import cv2
import numpy as np
import pytest
from seemore import SeemoReUpscaler
import torch

test_img_path = os.path.join(os.path.dirname(__file__), "bunny.jpeg")


def check_device(device: str):
    """Helper function to skip tests based on device availability"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available, skip test on cuda")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("mps is not available, skip test on mps")


def test_basic_upscaling():
    """测试基本的图像上采样功能"""
    seemore = SeemoReUpscaler("seemore_b_x4", device="cpu")
    img = cv2.imread(test_img_path)
    h, w = img.shape[:2]

    result = seemore(img)

    # 检查输出尺寸是否正确 (4x)
    assert result.shape[0] == h * 4
    assert result.shape[1] == w * 4
    assert result.shape[2] == 3  # BGR channels
    assert result.dtype == np.uint8


def test_image_modes():
    """测试不同的图像模式"""
    seemore = SeemoReUpscaler("seemore_b_x4", device="cpu")
    img = cv2.imread(test_img_path)

    # 测试灰度图像
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_result = seemore(gray_img)
    assert len(gray_result.shape) == 2  # 应该保持灰度

    # 测试 BGRA 图像
    bgra_img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    bgra_result = seemore(bgra_img)
    assert bgra_result.shape[2] == 4  # 应该保持 alpha 通道


def test_tile_inference():
    """测试分块推理"""
    seemore = SeemoReUpscaler("seemore_b_x4", device="cpu")
    img = cv2.imread(test_img_path)

    # 使用较小的 tile size 测试分块处理
    result_tiled = seemore(img, tile_size=64)
    result_full = seemore(img)

    # 检查分块处理的结果尺寸是否与完整处理相同
    assert result_tiled.shape == result_full.shape


def test_custom_scale():
    """测试自定义缩放比例"""
    seemore = SeemoReUpscaler("seemore_b_x4", device="cpu")
    img = cv2.imread(test_img_path)
    h, w = img.shape[:2]

    # 测试自定义 2x 缩放
    result = seemore(img, scale=2)
    assert result.shape[0] == h * 2
    assert result.shape[1] == w * 2


def test_invalid_model():
    """测试无效的模型名称"""
    with pytest.raises(ValueError):
        SeemoReUpscaler("invalid_model", device="cpu")


@pytest.mark.parametrize("device", ["cuda", "cpu", "mps"])
def test_device_upscaling(device):
    """Test upscaling on different devices"""
    check_device(device)
    seemore = SeemoReUpscaler("seemore_b_x4", device=device)
    img = cv2.imread(test_img_path)
    h, w = img.shape[:2]

    result = seemore(img)

    # Check output dimensions (4x)
    assert result.shape[0] == h * 4
    assert result.shape[1] == w * 4
    assert result.shape[2] == 3  # BGR channels
    assert result.dtype == np.uint8
