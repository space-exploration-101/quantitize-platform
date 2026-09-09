#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Golden E2E：496+281 全量流水线 + 检查。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "_lib") not in sys.path:
    sys.path.insert(0, str(_ROOT / "_lib"))

from job_config import JobConfig  # noqa: E402
from prepare_regression_job import populate_job  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checks import print_report, run_all_checks  # noqa: E402

REFERENCE = Path(__file__).resolve().parent / "reference" / "pt_eval_best_pt.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Golden E2E")
    parser.add_argument("--skip-pipeline", action="store_true")
    parser.add_argument("--freeze-baseline", action="store_true")
    parser.add_argument("--job-dir", type=Path, default=None, help="已有 job 目录（跳过 create/populate）")
    args = parser.parse_args()

    if args.job_dir:
        cfg = JobConfig.from_job_dir(args.job_dir)
    else:
        cfg = JobConfig.create_task("golden_regression", onnx_name="wrs_fp16_final2")
        populate_job(cfg, cali_count=496, test_count=281, use_copy=False)

        subprocess.run(
            [sys.executable, str(_ROOT / "07_validate_input.py"), "--job-dir", str(cfg.job_root)],
            check=True,
        )

    if not args.skip_pipeline:
        r = subprocess.run(
            [sys.executable, str(_ROOT / "05_runner.py"), "--job-dir", str(cfg.job_root)],
        )
        if r.returncode != 0:
            return r.returncode

    report = run_all_checks(cfg)
    print_report(report)

    e2e_path = cfg.results_dir / "e2e_test_report.json"
    e2e_path.parent.mkdir(parents=True, exist_ok=True)
    e2e_path.write_text(json.dumps({"job_id": cfg.job_id, **report}, indent=2), encoding="utf-8")

    summary = cfg.pt_eval_dir() / "summary.json"
    if args.freeze_baseline and summary.is_file():
        REFERENCE.parent.mkdir(parents=True, exist_ok=True)
        REFERENCE.write_text(summary.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Baseline frozen: {REFERENCE}")

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
