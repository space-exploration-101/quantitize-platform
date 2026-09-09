#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为 runner 编号脚本注入 _lib / pipeline / engine 路径。"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _is_contaminated(path: str) -> bool:
    p = path.lower().replace("\\", "/")
    return "isaac-sim" in p or "/isaacsim." in p


def scrub_isaac_from_environment() -> None:
    """去掉 Isaac Sim 注入的路径（其 pip_prebundle 与 yolov8/cp39 不兼容）。"""
    sys.path[:] = [p for p in sys.path if p and not _is_contaminated(p)]
    raw = os.environ.get("PYTHONPATH", "")
    if raw:
        kept = [p for p in raw.split(":") if p.strip() and not _is_contaminated(p)]
        if kept:
            os.environ["PYTHONPATH"] = ":".join(kept)
        else:
            os.environ.pop("PYTHONPATH", None)


def bootstrap_platform_script() -> Path:
    """返回 quantitize-platform 根目录。"""
    scrub_isaac_from_environment()

    lib_dir = Path(__file__).resolve().parent
    runner_dir = lib_dir.parent
    pipeline_dir = runner_dir.parent
    engine_dir = pipeline_dir / "engine"

    for p in (lib_dir, pipeline_dir, engine_dir):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)

    from std_platform import pin_stdlib_platform  # noqa: E402

    pin_stdlib_platform()

    from script_registry import PLATFORM_ROOT  # noqa: E402

    return PLATFORM_ROOT
