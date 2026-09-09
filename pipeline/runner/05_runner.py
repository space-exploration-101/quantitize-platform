#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""步骤 ⑧ 入口：8 步全流程编排。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_LIB = Path(__file__).resolve().parent / "_lib"
if str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))
from bootstrap import bootstrap_platform_script  # noqa: E402

bootstrap_platform_script()
from job_config import JobConfig  # noqa: E402
from runner_core import run_full_pipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="量化 Web 平台 — 8 步流水线")
    parser.add_argument("--job-dir", required=True, type=Path, help="jobs/{job_id} 目录")
    parser.add_argument("--onnx-name", default=None, help="覆盖 job_config 中的 onnx_name")
    args = parser.parse_args()
    overrides = {}
    if args.onnx_name:
        overrides["onnx_name"] = args.onnx_name
    cfg = JobConfig.from_job_dir(args.job_dir, **overrides)
    try:
        run_full_pipeline(cfg)
    except Exception as e:
        print(f"流水线失败: {e}", file=sys.stderr)
        return 1
    print(f"流水线完成: {cfg.job_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
