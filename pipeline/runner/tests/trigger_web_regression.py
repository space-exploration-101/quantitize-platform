#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Web 完成后自动 E2E 触发器。"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_TESTS = Path(__file__).resolve().parent
_LIB = _ROOT / "_lib"
if str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))
if str(_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_ROOT.parent))

from script_registry import OUTPUT_DATA_ROOT  # noqa: E402


def main() -> int:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"=== Auto regression {ts} ===")
    prep = subprocess.run(
        [
            sys.executable,
            str(_TESTS / "prepare_regression_job.py"),
            "--display-name",
            f"auto_regression_{ts}",
            "--onnx-name",
            "wrs_fp16_final2",
            "--task-id",
            "regression_auto",
        ],
    )
    if prep.returncode != 0:
        return prep.returncode

    job_dir = OUTPUT_DATA_ROOT / "regression_auto"
    return subprocess.run(
        [
            sys.executable,
            str(_TESTS / "run_golden_e2e.py"),
            "--job-dir",
            str(job_dir),
            "--freeze-baseline",
        ],
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
