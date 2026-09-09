#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""subprocess 环境：conda yolov8 与 PYTHONPATH（与 process_pipeline 对齐）。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

_PIPELINE = Path(__file__).resolve().parents[2]
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))

from script_registry import (  # noqa: E402
    ENGINE_DIR,
    PIPELINE_DIR,
    PLATFORM_ROOT,
    QUANTITIZE_DIR,
    QUANTITIZE_LEGACY_DIR,
    REPO_ROOT,
    RUNNER_LIB,
)

YOLOV8_ENV = "yolov8"


def _cudnn_lib_dirs() -> List[str]:
    """yolov8 等 conda 环境内 pip 安装的 nvidia-cudnn 库目录。"""
    bases: List[Path] = []
    if prefix := os.environ.get("CONDA_PREFIX"):
        bases.append(Path(prefix))
    home = Path.home()
    bases.extend(
        [
            home / "miniconda3" / "envs" / YOLOV8_ENV,
            home / "anaconda3" / "envs" / YOLOV8_ENV,
        ]
    )
    found: List[str] = []
    seen: set[str] = set()
    for base in bases:
        if not base.is_dir():
            continue
        for libdir in base.glob("lib/python*/site-packages/nvidia/cudnn/lib"):
            if not libdir.is_dir():
                continue
            key = str(libdir.resolve())
            if key in seen:
                continue
            if any(libdir.glob("libcudnn.so*")):
                seen.add(key)
                found.append(key)
    return found


def augment_gpu_env(env: dict) -> dict:
    """为 ONNX Runtime CUDA / PyTorch 补充 cuDNN 等动态库搜索路径。"""
    env = dict(env)
    prepend: List[str] = []
    for d in _cudnn_lib_dirs():
        if d not in prepend:
            prepend.append(d)
    if prepend:
        cur = env.get("LD_LIBRARY_PATH", "")
        parts = [p for p in cur.split(":") if p]
        for d in reversed(prepend):
            if d not in parts:
                parts.insert(0, d)
        env["LD_LIBRARY_PATH"] = ":".join(parts)
    return env


def apply_gpu_runtime_env() -> None:
    """在当前进程创建 ORT CUDA Session 前调用。"""
    augmented = augment_gpu_env(os.environ)
    for key, value in augmented.items():
        os.environ[key] = value


def _yolov8_python_candidates() -> List[Path]:
    """候选 yolov8 解释器路径（直接用二进制，避免 conda env list / conda run 在脏 PYTHONPATH 下失败）。"""
    home = Path.home()
    names = ("python3", "python")
    roots = [
        home / "miniconda3" / "envs" / YOLOV8_ENV / "bin",
        home / "anaconda3" / "envs" / YOLOV8_ENV / "bin",
        Path("/opt/conda/envs") / YOLOV8_ENV / "bin",
    ]
    out: List[Path] = []
    for root in roots:
        for name in names:
            out.append(root / name)
    return out


def resolve_yolov8_python() -> str:
    """返回 conda yolov8 的 python 绝对路径；找不到则报错（禁止回退 base）。"""
    if os.environ.get("CONDA_DEFAULT_ENV") == YOLOV8_ENV and sys.executable:
        exe = Path(sys.executable)
        if exe.is_file():
            return str(exe.resolve())
    override = os.environ.get("YOLOV8_PYTHON", "").strip()
    if override:
        p = Path(override).expanduser()
        if p.is_file():
            return str(p.resolve())
        raise RuntimeError(f"YOLOV8_PYTHON 无效: {override}")
    for cand in _yolov8_python_candidates():
        if cand.is_file():
            return str(cand.resolve())
    raise RuntimeError(
        f"未找到 conda 环境 {YOLOV8_ENV} 的 python。"
        f"请安装该环境，或设置 YOLOV8_PYTHON 指向其解释器。"
        f"已查找: {[str(p) for p in _yolov8_python_candidates()]}"
    )


def get_python_cmd() -> List[str]:
    """流水线子进程必须使用 yolov8；不再回退到当前/base python（量化会失败）。"""
    return [resolve_yolov8_python()]


def clean_pythonpath(env: dict) -> dict:
    """构造子进程 PYTHONPATH：runner/_lib + pipeline + engine；剔除易冲突根路径。"""
    env = dict(env)
    prefer = [str(RUNNER_LIB), str(PIPELINE_DIR), str(ENGINE_DIR)]
    raw = env.get("PYTHONPATH", "")
    paths: List[str] = []
    skip = {
        str(REPO_ROOT).rstrip("/"),
        str(PLATFORM_ROOT).rstrip("/"),
        str(QUANTITIZE_DIR).rstrip("/"),
        # 旧工作区若出现在环境里也跳过，避免混用两套代码
        str(QUANTITIZE_LEGACY_DIR).rstrip("/"),
    }
    for p in raw.split(":"):
        p = p.strip()
        if not p or "isaac-sim" in p.lower():
            continue
        if p.rstrip("/") in skip:
            continue
        paths.append(p)
    for lib in reversed(prefer):
        if lib not in paths:
            paths.insert(0, lib)
    env["PYTHONPATH"] = ":".join(paths)
    return env


def run_script(
    script_path: Path,
    args: List[str],
    *,
    cwd: Optional[Path] = None,
    log_path: Optional[Path] = None,
) -> subprocess.CompletedProcess:
    """在 conda yolov8 下执行 platform 脚本（与手动评测一致）。"""
    script_path = script_path.resolve()
    cmd = get_python_cmd() + [str(script_path), *args]
    env = augment_gpu_env(clean_pythonpath(os.environ.copy()))
    cwd = cwd or script_path.parent
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(cwd),
        env=env,
        encoding="utf-8",
        errors="replace",
    )
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"$ {' '.join(cmd)}\n")
            f.write(f"cwd={cwd}\n")
            if result.stdout:
                f.write(result.stdout)
            if result.stderr:
                f.write(result.stderr)
    return result
