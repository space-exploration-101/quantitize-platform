#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一黑白图预处理（WEB_DESIGN.md §4.2）。

路径 A：1280 灰度三通道 — PT / 量化 ONNX 测试
路径 B：1280 → 2000 png2bin → bin2png → resize 1280 — FPGA 侧视 ONNX 测试
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
    bgr_hwc_to_chw_tensor,
    grayscale_r_channel_bgr,
    grayscale_to_chw_tensor,
    imread_unicode,
    preprocess_bgr_by_mode,
    preprocess_by_mode,
    preprocess_grayscale_path_a,
    preprocess_grayscale_r_channel_from_bgr,
    preprocess_rgb_from_bgr,
    to_grayscale_hw,
)
from script_registry import EngineScripts, engine_script  # noqa: E402


class PreprocessPath(str, Enum):
    A_DIRECT_1280 = "A"
    B_FPGA_ROUNDTRIP = "B"


class InputPreprocessMode(str, Enum):
    """推理/标定输入预处理（与 WEB 创建任务选项一致）。"""

    RGB = "rgb"
    GRAYSCALE_UNIFORM = "grayscale_uniform"
    GRAYSCALE_R_CHANNEL = "grayscale_r_channel"


INPUT_PREPROCESS_MODE_LABELS = {
    InputPreprocessMode.RGB: "RGB / 彩色预训练（原 BGR resize）",
    InputPreprocessMode.GRAYSCALE_UNIFORM: "黑白 — 三通道同值（统一灰度标定）",
    InputPreprocessMode.GRAYSCALE_R_CHANNEL: "黑白 — R 通道（脚本 / readme 格式）",
}


def normalize_preprocess_mode(mode: str) -> str:
    m = (mode or InputPreprocessMode.GRAYSCALE_UNIFORM.value).strip()
    if m not in PREPROCESS_MODES:
        raise ValueError(f"无效 preprocess_mode: {mode!r}，可选 {PREPROCESS_MODES}")
    return m


def bgr_for_pt_predict(img_bgr: np.ndarray, mode: str, imgsz: int) -> np.ndarray:
    """Ultralytics predict 用 BGR uint8 图。"""
    mode = normalize_preprocess_mode(mode)
    if mode == InputPreprocessMode.RGB.value:
        return cv2.resize(img_bgr, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    if mode == InputPreprocessMode.GRAYSCALE_R_CHANNEL.value:
        return grayscale_r_channel_bgr(img_bgr, (imgsz, imgsz))
    gray = to_grayscale_hw(img_bgr)
    gray_r = cv2.resize(gray, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(gray_r, cv2.COLOR_GRAY2BGR)


def preprocess_path_a(
    image_path: str | Path,
    target_size: Tuple[int, int] = (1280, 1280),
    dtype=np.float16,
) -> np.ndarray:
    return preprocess_grayscale_path_a(image_path, target_size, dtype=dtype)


def preprocess_path_b(
    image_path: str | Path,
    target_size: Tuple[int, int] = (1280, 1280),
    fpga_size: int = 2000,
    dtype=np.float16,
) -> Tuple[np.ndarray, Path]:
    import tempfile

    from png_bin_converter import bin_to_png, png_to_bin  # noqa: WPS433

    image_path = Path(image_path)
    img = imread_unicode(image_path)
    gray = to_grayscale_hw(img)
    h128, w128 = target_size
    gray128 = cv2.resize(gray, (w128, h128), interpolation=cv2.INTER_LINEAR)
    tmp_dir = Path(tempfile.mkdtemp(prefix="fpga_pre_"))
    src_png = tmp_dir / f"{image_path.stem}_1280gray.png"
    bgr = cv2.cvtColor(gray128, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(src_png), bgr)

    bin_path = tmp_dir / f"{image_path.stem}.bin"
    side_view_path = tmp_dir / f"{image_path.stem}_side_view.png"
    png_to_bin(str(src_png), str(bin_path), target_size=fpga_size)
    bin_to_png(str(bin_path), str(side_view_path), width=fpga_size, height=fpga_size)

    side = cv2.imread(str(side_view_path), cv2.IMREAD_GRAYSCALE)
    if side is None:
        raise ValueError(f"侧视图读取失败: {side_view_path}")
    tensor = grayscale_to_chw_tensor(side, target_size, dtype=dtype)
    return tensor, side_view_path


def converter_script_path() -> Path:
    return engine_script(EngineScripts.PNG_BIN_CONVERTER)
