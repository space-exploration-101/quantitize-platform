#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一输入预处理。

量化模型 ABI 固定为 NCHW1：[1, 1, 1280, 1280] FP16。
路径 A：源图 → 1280 gray1
路径 B：1280 → 2000 png2bin → bin2png → resize 1280（FPGA 侧视）
"""

from __future__ import annotations

import sys
from enum import Enum
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

_PIPELINE_DIR = Path(__file__).resolve().parents[2]
_ENGINE_DIR = _PIPELINE_DIR / "engine"
for _p in (_PIPELINE_DIR, _ENGINE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from grayscale_preprocess import (  # noqa: E402
    PREPROCESS_MODES,
    gray1_to_chw_tensor,
    imread_unicode,
    normalize_preprocess_mode,
    plane_color_to_gray,
    plane_passthrough,
    preprocess_bgr_by_mode,
    preprocess_by_mode,
    to_grayscale_hw,
)

__all__ = [
    "PREPROCESS_MODES",
    "InputPreprocessMode",
    "INPUT_PREPROCESS_MODE_LABELS",
    "normalize_preprocess_mode",
    "bgr_for_pt_predict",
    "imread_unicode",
    "preprocess_bgr_by_mode",
    "preprocess_by_mode",
]
from script_registry import EngineScripts, engine_script  # noqa: E402


class PreprocessPath(str, Enum):
    A_DIRECT_1280 = "A"
    B_FPGA_ROUNDTRIP = "B"


class InputPreprocessMode(str, Enum):
    PASSTHROUGH = "passthrough"
    COLOR_TO_GRAY = "color_to_gray"


INPUT_PREPROCESS_MODE_LABELS = {
    InputPreprocessMode.PASSTHROUGH: "不做处理（已是灰度/R 通道，输出 1 通道）",
    InputPreprocessMode.COLOR_TO_GRAY: "彩图转灰度（BGR2GRAY，输出 1 通道）",
}


def bgr_for_pt_predict(img_bgr: np.ndarray, mode: str, imgsz: int) -> np.ndarray:
    """Ultralytics 1-channel predict: HWC1 uint8."""
    mode = normalize_preprocess_mode(mode)
    plane = plane_passthrough(img_bgr) if mode == InputPreprocessMode.PASSTHROUGH.value else plane_color_to_gray(img_bgr)
    resized = cv2.resize(plane, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    return resized[:, :, np.newaxis]


def preprocess_path_a(
    image_path: str | Path,
    target_size: Tuple[int, int] = (1280, 1280),
    dtype=np.float16,
) -> np.ndarray:
    return preprocess_by_mode(image_path, "color_to_gray", target_size, dtype=dtype)


def preprocess_path_b(
    image_path: str | Path,
    target_size: Tuple[int, int] = (1280, 1280),
    fpga_size: int = 2000,
    dtype=np.float16,
    preprocess_mode: str = "passthrough",
) -> Tuple[np.ndarray, Path]:
    import tempfile

    from png_bin_converter import bin_to_png, png_to_bin  # noqa: WPS433

    image_path = Path(image_path)
    mode = normalize_preprocess_mode(preprocess_mode)
    tmp_dir = Path(tempfile.mkdtemp(prefix="fpga_pre_"))
    src_png = tmp_dir / f"{image_path.stem}_1280gray.png"
    bin_path = tmp_dir / f"{image_path.stem}.bin"
    side_view_path = tmp_dir / f"{image_path.stem}_side_view.png"

    img = imread_unicode(image_path, flags=int(cv2.IMREAD_UNCHANGED))
    plane = plane_passthrough(img) if mode == "passthrough" else plane_color_to_gray(img)
    h128, w128 = target_size
    gray128 = cv2.resize(plane, (w128, h128), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(str(src_png), gray128)

    png_to_bin(str(src_png), str(bin_path), target_size=fpga_size, preprocess_mode=mode)
    bin_to_png(str(bin_path), str(side_view_path), width=fpga_size, height=fpga_size, preprocess_mode=mode)

    side = cv2.imread(str(side_view_path), cv2.IMREAD_UNCHANGED)
    if side is None:
        raise ValueError(f"侧视图读取失败: {side_view_path}")
    tensor = gray1_to_chw_tensor(side, target_size, dtype=dtype)
    return tensor, side_view_path


def converter_script_path() -> Path:
    return engine_script(EngineScripts.PNG_BIN_CONVERTER)
