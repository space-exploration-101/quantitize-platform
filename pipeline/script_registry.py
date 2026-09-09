#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
流水线脚本路径注册表（quantitize-platform）。

路径单一事实来源：本模块。数据目录只认环境变量或本文件位置，不使用 cwd。

- engine  → pipeline/engine/
- runner  → pipeline/runner/
- patches → patches/
- data    → data/shared_data, data/output_data
"""

from __future__ import annotations

import os
from pathlib import Path


def _abs_from_env(name: str, default: Path, *, base: Path) -> Path:
    """解析目录：环境变量优先；相对路径相对 base，绝不相对 cwd。"""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default.resolve()
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = base / p
    return p.resolve()


# 代码位置（本文件所在产品树），与 cwd 无关
_CODE_ROOT = Path(__file__).resolve().parents[1]

# 产品根：可被 PLATFORM_ROOT 覆盖（容器或测试隔离）
PLATFORM_ROOT = _abs_from_env("PLATFORM_ROOT", _CODE_ROOT, base=_CODE_ROOT)

# 脚本始终跟代码走，避免只改 PLATFORM_ROOT 时找不到 engine/runner
ENGINE_DIR = _CODE_ROOT / "pipeline" / "engine"
RUNNER_DIR = _CODE_ROOT / "pipeline" / "runner"
RUNNER_LIB = RUNNER_DIR / "_lib"
PIPELINE_DIR = _CODE_ROOT / "pipeline"
PATCHES_DIR = _CODE_ROOT / "patches"

DATA_DIR = PLATFORM_ROOT / "data"
OUTPUT_DATA_ROOT = _abs_from_env(
    "OUTPUT_DATA_ROOT", DATA_DIR / "output_data", base=PLATFORM_ROOT
)
SHARED_DATA_ROOT = _abs_from_env(
    "SHARED_DATA_ROOT", DATA_DIR / "shared_data", base=PLATFORM_ROOT
)

# Optional high-speed per-task scratch.  Empty means the legacy all-in-job-root
# layout.  New tasks persist whether scratch is enabled in job_config.json so
# historical jobs keep their original layout.
_scratch_raw = os.environ.get("TASK_SCRATCH_ROOT", "").strip()
TASK_SCRATCH_ROOT = (
    _abs_from_env("TASK_SCRATCH_ROOT", PLATFORM_ROOT / ".scratch", base=PLATFORM_ROOT)
    if _scratch_raw
    else None
)

# 兼容旧 _lib 命名（历史代码里 QUANTITIZE_DIR = 产品根）
QUANTITIZE_DIR = PLATFORM_ROOT
# 用于从 PYTHONPATH 剔除的「仓库上层」
REPO_ROOT = PLATFORM_ROOT.parent

# 旧项目只读参考（灌数据集 / 回归素材）。默认是平台目录的兄妹 quantitize/。
QUANTITIZE_LEGACY_DIR = _abs_from_env(
    "QUANTITIZE_LEGACY_DIR",
    PLATFORM_ROOT.parent / "quantitize",
    base=PLATFORM_ROOT,
)


class EngineScripts:
    """引擎层 — 量化与 bin 合并。"""

    QUANTIZE = "quantitize.py"
    GENERATE_LAYER_BIN = "check_certain_layer_multi.py"
    RENAME_WT = "01_rename_wt_files.py"
    MERGE_WT = "02_merge_wt_files.py"
    RENAME_BN = "03_rename_bn_files.py"
    MERGE_BN = "04_merge_bn.py"
    PNG_BIN_CONVERTER = "png_bin_converter.py"
    SIMPLE_ONNX_INFERENCE = "simple_onnx_inference.py"


class PlatformScripts:
    """编排层 — pipeline/runner。"""

    PT_EVAL = "01_pt_eval.py"
    ONNX_EVAL = "02_onnx_eval.py"
    YOLO_LABELS_TO_JSON = "03_yolo_labels_to_json.py"
    GENERATE_FPGA_TEST_PACK = "04_generate_fpga_test_pack.py"
    RUNNER = "05_runner.py"
    BUNDLE = "06_bundle.py"
    VALIDATE_INPUT = "07_validate_input.py"
    ARCHIVE_JOBS = "10_archive_jobs.py"
    EVAL_FPGA_RESULTS = "eval_fpga_results.py"


def engine_script(name: str) -> Path:
    path = ENGINE_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"引擎脚本不存在: {path}")
    return path


def platform_script(name: str) -> Path:
    path = RUNNER_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"编排脚本不存在: {path}")
    return path


def all_registered_scripts() -> dict:
    out = {}
    for key, val in vars(EngineScripts).items():
        if key.startswith("_") or not isinstance(val, str):
            continue
        out[f"engine.{key}"] = str(engine_script(val))
    for key, val in vars(PlatformScripts).items():
        if key.startswith("_") or not isinstance(val, str):
            continue
        out[f"platform.{key}"] = str(platform_script(val))
    return out


if __name__ == "__main__":
    print(f"PLATFORM_ROOT={PLATFORM_ROOT}")
    print(f"ENGINE_DIR={ENGINE_DIR}")
    print(f"RUNNER_DIR={RUNNER_DIR}")
    print(f"DATA_DIR={DATA_DIR}")
    print(f"OUTPUT_DATA_ROOT={OUTPUT_DATA_ROOT}")
    print(f"SHARED_DATA_ROOT={SHARED_DATA_ROOT}")
    print(f"TASK_SCRATCH_ROOT={TASK_SCRATCH_ROOT or ''}")
    print(f"PATCHES_DIR={PATCHES_DIR}")
    print(f"QUANTITIZE_LEGACY_DIR={QUANTITIZE_LEGACY_DIR}")
    for name, path in sorted(all_registered_scripts().items()):
        print(f"{name}\t{path}")
