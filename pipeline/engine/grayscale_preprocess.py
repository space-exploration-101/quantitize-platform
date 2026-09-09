#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一输入预处理（引擎层，供 quantitize.py 与 platform 共用）。

注意：模型推理仍然需要 resize、BGR->RGB、/255、NCHW 这些格式转换。
这里的“预处理模式”主要指是否额外改动图像通道/灰度内容。
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def imread_unicode(path: str | Path) -> np.ndarray:
    path = str(path)
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图片: {path}")
    return img


def to_grayscale_hw(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.ndim == 2:
        return img_bgr
    if img_bgr.shape[2] == 1:
        return img_bgr[:, :, 0]
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def grayscale_to_chw_tensor(
    gray_hw: np.ndarray,
    target_size: Tuple[int, int] = (1280, 1280),
    dtype=np.float16,
) -> np.ndarray:
    """灰度三通道同值，/255，(1, 3, H, W)。"""
    h, w = target_size
    resized = cv2.resize(gray_hw, (w, h), interpolation=cv2.INTER_LINEAR)
    chw = np.stack([resized, resized, resized], axis=0).astype(np.float32) / 255.0
    return np.expand_dims(chw, axis=0).astype(dtype)


def preprocess_grayscale_path_a(
    image_path: str | Path,
    target_size: Tuple[int, int] = (1280, 1280),
    dtype=np.float16,
) -> np.ndarray:
    img = imread_unicode(image_path)
    gray = to_grayscale_hw(img)
    return grayscale_to_chw_tensor(gray, target_size, dtype=dtype)


def preprocess_grayscale_from_bgr(
    img_bgr: np.ndarray,
    target_size: Tuple[int, int] = (1280, 1280),
    dtype=np.float16,
) -> np.ndarray:
    gray = to_grayscale_hw(img_bgr)
    return grayscale_to_chw_tensor(gray, target_size, dtype=dtype)


def bgr_hwc_to_chw_tensor(
    bgr_hwc: np.ndarray,
    dtype=np.float16,
) -> np.ndarray:
    """BGR HWC → RGB NCHW，/255。"""
    rgb = bgr_hwc[..., ::-1]
    chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(chw, axis=0).astype(dtype)


def preprocess_rgb_path(
    image_path: str | Path,
    target_size: Tuple[int, int] = (1280, 1280),
    dtype=np.float16,
) -> np.ndarray:
    """RGB/彩色预训练：BGR resize，不强制灰度，转 RGB NCHW。"""
    img = imread_unicode(image_path)
    h, w = target_size
    resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    return bgr_hwc_to_chw_tensor(resized, dtype=dtype)


def preprocess_rgb_from_bgr(
    img_bgr: np.ndarray,
    target_size: Tuple[int, int] = (1280, 1280),
    dtype=np.float16,
) -> np.ndarray:
    h, w = target_size
    resized = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    return bgr_hwc_to_chw_tensor(resized, dtype=dtype)


def preprocess_passthrough_path(
    image_path: str | Path,
    target_size: Tuple[int, int] = (1280, 1280),
    dtype=np.float16,
) -> np.ndarray:
    """保持原图通道内容，不额外转灰度。

    适用于输入文件本身已经按模型训练方式准备好的情况，例如：
    - RGB 彩色图：保持 RGB 内容；
    - R-only 图：保持 R=gray, G=B=0，不再执行 BGR2GRAY，避免亮度被 0.299 权重压暗。

    仍会执行模型必需的格式处理：resize、BGR->RGB、/255、NCHW。
    """
    return preprocess_rgb_path(image_path, target_size, dtype=dtype)


def preprocess_passthrough_from_bgr(
    img_bgr: np.ndarray,
    target_size: Tuple[int, int] = (1280, 1280),
    dtype=np.float16,
) -> np.ndarray:
    """保持已读入图像的原始通道内容，不额外转灰度。"""
    return preprocess_rgb_from_bgr(img_bgr, target_size, dtype=dtype)


def grayscale_r_channel_bgr(
    img_bgr: np.ndarray,
    target_size: Tuple[int, int] = (1280, 1280),
) -> np.ndarray:
    """脚本/readme 格式：BGR2GRAY 后仅 R 通道有灰度（B=G=0）。"""
    gray = to_grayscale_hw(img_bgr)
    h, w = target_size
    gray_r = cv2.resize(gray, (w, h), interpolation=cv2.INTER_LINEAR)
    bgr = np.zeros((h, w, 3), dtype=np.uint8)
    bgr[:, :, 2] = gray_r
    return bgr


def preprocess_grayscale_r_channel(
    image_path: str | Path,
    target_size: Tuple[int, int] = (1280, 1280),
    dtype=np.float16,
) -> np.ndarray:
    img = imread_unicode(image_path)
    bgr = grayscale_r_channel_bgr(img, target_size)
    return bgr_hwc_to_chw_tensor(bgr, dtype=dtype)


def preprocess_grayscale_r_channel_from_bgr(
    img_bgr: np.ndarray,
    target_size: Tuple[int, int] = (1280, 1280),
    dtype=np.float16,
) -> np.ndarray:
    bgr = grayscale_r_channel_bgr(img_bgr, target_size)
    return bgr_hwc_to_chw_tensor(bgr, dtype=dtype)


PREPROCESS_MODES = ("rgb", "grayscale_uniform", "grayscale_r_channel", "passthrough")


def preprocess_by_mode(
    image_path: str | Path,
    mode: str,
    target_size: Tuple[int, int] = (1280, 1280),
    dtype=np.float16,
) -> np.ndarray:
    if mode in ("rgb", "passthrough"):
        return preprocess_rgb_path(image_path, target_size, dtype=dtype)
    if mode == "grayscale_r_channel":
        return preprocess_grayscale_r_channel(image_path, target_size, dtype=dtype)
    if mode != "grayscale_uniform":
        raise ValueError(f"未知 preprocess_mode: {mode!r}，可选 {PREPROCESS_MODES}")
    return preprocess_grayscale_path_a(image_path, target_size, dtype=dtype)


def preprocess_bgr_by_mode(
    img_bgr: np.ndarray,
    mode: str,
    target_size: Tuple[int, int] = (1280, 1280),
    dtype=np.float16,
) -> np.ndarray:
    if mode in ("rgb", "passthrough"):
        return preprocess_rgb_from_bgr(img_bgr, target_size, dtype=dtype)
    if mode == "grayscale_r_channel":
        return preprocess_grayscale_r_channel_from_bgr(img_bgr, target_size, dtype=dtype)
    if mode != "grayscale_uniform":
        raise ValueError(f"未知 preprocess_mode: {mode!r}，可选 {PREPROCESS_MODES}")
    return preprocess_grayscale_from_bgr(img_bgr, target_size, dtype=dtype)
