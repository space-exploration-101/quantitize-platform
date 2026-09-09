#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""固定加载标准库 platform，避免 cwd/脚本目录下的本地 package「platform/」遮蔽它。

引擎脚本（quantitize.py 等）必须在 import torch / ultralytics / onnxruntime 之前：
    import _pin_stdlib_platform  # noqa: F401
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def pin_stdlib_platform() -> None:
    mod = sys.modules.get("platform")
    if mod is not None and hasattr(mod, "system"):
        return
    ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    for base in (sys.base_prefix, sys.prefix):
        py = Path(base) / "lib" / f"python{ver}" / "platform.py"
        if py.is_file():
            spec = importlib.util.spec_from_file_location("platform", py)
            if spec and spec.loader:
                loaded = importlib.util.module_from_spec(spec)
                sys.modules["platform"] = loaded
                spec.loader.exec_module(loaded)
                return
    raise ImportError("无法加载标准库 platform 模块")


pin_stdlib_platform()
