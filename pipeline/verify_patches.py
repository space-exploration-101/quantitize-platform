#!/usr/bin/env python3
"""M2 冒烟：确认 patches 被加载到 sys.modules（T-P1）。"""

from __future__ import annotations

import sys
from pathlib import Path

ENGINE = Path(__file__).resolve().parent / "engine"
sys.path.insert(0, str(ENGINE))

import _pin_stdlib_platform  # noqa: E402, F401
import _load_local_onnxruntime_modules  # noqa: E402

import onnxruntime as ort  # noqa: E402
from onnxruntime.quantization import quant_utils  # noqa: E402
from onnxruntime.quantization.operators import conv  # noqa: E402


def main() -> int:
    qu = Path(quant_utils.__file__).resolve()
    cv = Path(conv.__file__).resolve()
    print("ort", ort.__version__, ort.__file__)
    print("quant_utils", qu)
    print("conv", cv)
    print("providers", ort.get_available_providers())
    ok = "/patches/onnxruntime/" in str(qu).replace("\\", "/") and "/patches/onnxruntime/" in str(
        cv
    ).replace("\\", "/")
    if not ok:
        print("FAIL: modules not loaded from patches/")
        return 1
    print("PASS: patches loaded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
