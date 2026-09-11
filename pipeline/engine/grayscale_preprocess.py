#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gray1 输入预处理：量化模型始终吃 [1, 1, H, W]。

网页只暴露两类通道处理：
- passthrough / 不做处理：已是灰度或 R-only 时，不跑 BGR2GRAY（避免 0.299 压暗）
- color_to_gray / 彩图转灰度：OpenCV BGR2GRAY

resize、/255、NCHW 是模型格式要求，两种模式都会做。
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

PREPROCESS_MODES = ("passthrough", "color_to_gray")
PREPROCESS_MODE_ALIASES = {
    "passthrough": "passthrough",
    "color_to_gray": "color_to_gray",
    "gray1": "color_to_gray",
    "rgb": "color_to_gray",
    "grayscale_uniform": "color_to_gray",
    "grayscale_r_channel": "color_to_gray",
}


def normalize_preprocess_mode(mode: str | None) -> str:
    raw = (mode or "passthrough").strip()
    mapped = PREPROCESS_MODE_ALIASES.get(raw, raw)
    if mapped not in PREPROCESS_MODES:
        raise ValueError(f"未知 preprocess_mode: {mode!r}，可选 {PREPROCESS_MODES}")
    return mapped


def imread_unicode(path: str | Path, flags: int = int(cv2.IMREAD_COLOR)) -> np.ndarray:
    path = str(path)
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    if img is None:
        raise ValueError(f"无法读取图片: {path}")
    return img


def _as_hw(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] == 1:
        return img[:, :, 0]
    raise ValueError(f"Expected HW or HWC1 image, got shape {img.shape}")


def plane_passthrough(img: np.ndarray) -> np.ndarray:
    """Keep stored intensity. 1-channel as-is; 3-channel takes R, not BGR2GRAY."""
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] == 1:
        return img[:, :, 0]
    if img.ndim == 3 and img.shape[2] >= 3:
        return img[:, :, 2]
    raise ValueError(f"Unsupported image shape for passthrough: {img.shape}")


def plane_color_to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] == 1:
        return img[:, :, 0]
    if img.ndim == 3 and img.shape[2] >= 3:
        return cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unsupported image shape for color_to_gray: {img.shape}")


def to_grayscale_hw(img_bgr: np.ndarray) -> np.ndarray:
    """Legacy helper used by FPGA packing; prefers a true gray plane."""
    return plane_color_to_gray(img_bgr)


def gray1_to_chw_tensor(
    gray_hw: np.ndarray,
    target_size: Tuple[int, int] = (1280, 1280),
    dtype=np.float16,
) -> np.ndarray:
    """Resize one grayscale plane and return normalized NCHW1 data."""
    h, w = target_size
    plane = _as_hw(gray_hw)
    resized = cv2.resize(plane, (w, h), interpolation=cv2.INTER_LINEAR)
    chw = resized[np.newaxis, ...].astype(np.float32) / 255.0
    return np.expand_dims(chw, axis=0).astype(dtype)


def grayscale_to_chw_tensor(
    gray_hw: np.ndarray,
    target_size: Tuple[int, int] = (1280, 1280),
    dtype=np.float16,
) -> np.ndarray:
    """Compatibility name: now always NCHW1, not replicated RGB."""
    return gray1_to_chw_tensor(gray_hw, target_size, dtype=dtype)


def _load_plane(image_path: str | Path, mode: str) -> np.ndarray:
    mode = normalize_preprocess_mode(mode)
    if mode == "passthrough":
        img = imread_unicode(image_path, flags=int(cv2.IMREAD_UNCHANGED))
        return plane_passthrough(img)
    img = imread_unicode(image_path, flags=int(cv2.IMREAD_COLOR))
    return plane_color_to_gray(img)


def preprocess_by_mode(
    image_path: str | Path,
    mode: str,
    target_size: Tuple[int, int] = (1280, 1280),
    dtype=np.float16,
) -> np.ndarray:
    return gray1_to_chw_tensor(_load_plane(image_path, mode), target_size, dtype=dtype)


def preprocess_bgr_by_mode(
    img_bgr: np.ndarray,
    mode: str,
    target_size: Tuple[int, int] = (1280, 1280),
    dtype=np.float16,
) -> np.ndarray:
    mode = normalize_preprocess_mode(mode)
    plane = plane_passthrough(img_bgr) if mode == "passthrough" else plane_color_to_gray(img_bgr)
    return gray1_to_chw_tensor(plane, target_size, dtype=dtype)


def preprocess_gray1_path(
    image_path: str | Path,
    target_size: Tuple[int, int] = (1280, 1280),
    dtype=np.float16,
) -> np.ndarray:
    return preprocess_by_mode(image_path, "color_to_gray", target_size, dtype=dtype)


def preprocess_gray1_from_bgr(
    img_bgr: np.ndarray,
    target_size: Tuple[int, int] = (1280, 1280),
    dtype=np.float16,
) -> np.ndarray:
    return preprocess_bgr_by_mode(img_bgr, "color_to_gray", target_size, dtype=dtype)


def preprocess_passthrough_path(
    image_path: str | Path,
    target_size: Tuple[int, int] = (1280, 1280),
    dtype=np.float16,
) -> np.ndarray:
    return preprocess_by_mode(image_path, "passthrough", target_size, dtype=dtype)


def preprocess_passthrough_from_bgr(
    img_bgr: np.ndarray,
    target_size: Tuple[int, int] = (1280, 1280),
    dtype=np.float16,
) -> np.ndarray:
    return preprocess_bgr_by_mode(img_bgr, "passthrough", target_size, dtype=dtype)


def preprocess_grayscale_path_a(
    image_path: str | Path,
    target_size: Tuple[int, int] = (1280, 1280),
    dtype=np.float16,
) -> np.ndarray:
    return preprocess_by_mode(image_path, "color_to_gray", target_size, dtype=dtype)


def preprocess_grayscale_from_bgr(
    img_bgr: np.ndarray,
    target_size: Tuple[int, int] = (1280, 1280),
    dtype=np.float16,
) -> np.ndarray:
    return preprocess_bgr_by_mode(img_bgr, "color_to_gray", target_size, dtype=dtype)


def bgr_hwc_to_chw_tensor(
    bgr_hwc: np.ndarray,
    dtype=np.float16,
) -> np.ndarray:
    """Deprecated 3-channel helper; kept for old imports, converts to NCHW1 via R."""
    return gray1_to_chw_tensor(plane_passthrough(bgr_hwc), (bgr_hwc.shape[0], bgr_hwc.shape[1]), dtype=dtype)


def preprocess_rgb_path(
    image_path: str | Path,
    target_size: Tuple[int, int] = (1280, 1280),
    dtype=np.float16,
) -> np.ndarray:
    return preprocess_by_mode(image_path, "color_to_gray", target_size, dtype=dtype)


def preprocess_rgb_from_bgr(
    img_bgr: np.ndarray,
    target_size: Tuple[int, int] = (1280, 1280),
    dtype=np.float16,
) -> np.ndarray:
    return preprocess_bgr_by_mode(img_bgr, "color_to_gray", target_size, dtype=dtype)


def grayscale_r_channel_bgr(
    img_bgr: np.ndarray,
    target_size: Tuple[int, int] = (1280, 1280),
) -> np.ndarray:
    """PT helper: return HWC1 gray for 1-channel Ultralytics models."""
    h, w = target_size
    gray = cv2.resize(plane_color_to_gray(img_bgr), (w, h), interpolation=cv2.INTER_LINEAR)
    return gray[:, :, np.newaxis]


def preprocess_grayscale_r_channel(
    image_path: str | Path,
    target_size: Tuple[int, int] = (1280, 1280),
    dtype=np.float16,
) -> np.ndarray:
    return preprocess_by_mode(image_path, "color_to_gray", target_size, dtype=dtype)


def preprocess_grayscale_r_channel_from_bgr(
    img_bgr: np.ndarray,
    target_size: Tuple[int, int] = (1280, 1280),
    dtype=np.float16,
) -> np.ndarray:
    return preprocess_bgr_by_mode(img_bgr, "color_to_gray", target_size, dtype=dtype)
