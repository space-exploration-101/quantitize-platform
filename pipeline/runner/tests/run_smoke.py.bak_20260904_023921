#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke 测试：40 cali + 20 test，跑通 8 步并检查产物结构。"""

from __future__ import annotations

import argparse
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke regression")
    parser.add_argument("--skip-pipeline", action="store_true")
    parser.add_argument("--job-dir", type=Path, default=None, help="已有 job 目录（仅跑 checks）")
    args = parser.parse_args()

    if args.job_dir:
        cfg = JobConfig.from_job_dir(args.job_dir)
    else:
        cfg = JobConfig.create_task("smoke_test", onnx_name="smoke")
        populate_job(cfg, cali_count=40, test_count=20, use_copy=False)
        cfg.min_cali_images = 40
        cfg.min_test_images = 20
        cfg.save()

        subprocess.run(
            [
                sys.executable,
                str(_ROOT / "07_validate_input.py"),
                "--job-dir",
                str(cfg.job_root),
                "--min-cali",
                "40",
                "--min-test",
                "20",
            ],
            check=False,
        )

        sys.path.insert(0, str(_ROOT / "_lib"))
        from input_manifest import write_input_manifest  # noqa: E402

        write_input_manifest(cfg, validation_ok=True)

    if not args.skip_pipeline and not args.job_dir:
        r = subprocess.run(
            [sys.executable, str(_ROOT / "05_runner.py"), "--job-dir", str(cfg.job_root)],
        )
        if r.returncode != 0:
            print("Pipeline failed", file=sys.stderr)
            return r.returncode

    report = run_all_checks(cfg)
    print_report(report)
    out = cfg.results_dir / "smoke_check_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    import json
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
