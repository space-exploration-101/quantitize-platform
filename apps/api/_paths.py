#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把新平台 pipeline 放进 sys.path；不引入旧 quantitize/ 运行时。"""

from __future__ import annotations

import sys
from pathlib import Path

# apps/api/_paths.py → quantitize-platform/
PLATFORM_ROOT = Path(__file__).resolve().parents[2]


def setup_pipeline_imports() -> Path:
    pipeline = PLATFORM_ROOT / "pipeline"
    for p in (
        pipeline / "runner" / "_lib",
        pipeline,
        pipeline / "engine",
    ):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    return PLATFORM_ROOT
