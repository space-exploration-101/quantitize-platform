#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检校 jobs/{id}/input 上传数据，输出 JSON 报告（Web 上传后调用）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_LIB = Path(__file__).resolve().parent / "_lib"
if str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))
from bootstrap import bootstrap_platform_script  # noqa: E402

bootstrap_platform_script()
from job_config import JobConfig  # noqa: E402
from validate_input import save_report, validate_job_input  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="检校任务上传数据")
    parser.add_argument("--job-dir", required=True, type=Path)
    parser.add_argument("--min-cali", type=int, default=400)
    parser.add_argument("--min-test", type=int, default=100)
    parser.add_argument("-o", "--output", type=Path, default=None, help="validation_report.json")
    args = parser.parse_args()

    cfg = JobConfig.from_job_dir(args.job_dir)
    report = validate_job_input(
        cfg.job_root,
        min_cali=args.min_cali,
        min_test=args.min_test,
        nc=cfg.nc,
        expect_imgsz=cfg.imgsz,
    )
    out = args.output or (cfg.job_root / "input_validation.json")
    save_report(report, out)

    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    if not report.ok:
        print(f"\n检校失败: {len(report.errors)} 个错误", file=sys.stderr)
        return 1
    print(f"\n检校通过（{len(report.warnings)} 条警告）→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
