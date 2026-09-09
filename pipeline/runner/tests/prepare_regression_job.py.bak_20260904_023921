#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""组装 regression job 到 output_data/。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_LIB = _ROOT / "_lib"
if str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))
from bootstrap import bootstrap_platform_script  # noqa: E402
from script_registry import QUANTITIZE_LEGACY_DIR  # noqa: E402

QUANTITIZE_DIR = bootstrap_platform_script()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from job_config import JobConfig  # noqa: E402

SOURCE_PT = QUANTITIZE_LEGACY_DIR / "best.pt"
SOURCE_CALI = QUANTITIZE_LEGACY_DIR / "cali_data"
SOURCE_QT_IMG = QUANTITIZE_LEGACY_DIR / "Quantitative test" / "quantize_test" / "images" / "test"
SOURCE_QT_LAB = QUANTITIZE_LEGACY_DIR / "Quantitative test" / "quantize_test" / "labels" / "test"


def link_or_copy(src: Path, dst: Path, use_copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if use_copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def populate_job(
    cfg: JobConfig,
    *,
    cali_count: int = 496,
    test_count: int = 281,
    use_copy: bool = False,
) -> None:
    inp = cfg.input_dir
    cali_out = cfg.cali_dir
    img_out = cfg.test_images_dir
    lab_out = cfg.test_labels_dir
    for d in (cali_out, img_out, lab_out):
        d.mkdir(parents=True, exist_ok=True)
    link_or_copy(SOURCE_PT, inp / "model.pt", use_copy)
    for cf in sorted(SOURCE_CALI.glob("*.png"))[:cali_count]:
        link_or_copy(cf, cali_out / cf.name, use_copy)
    test_imgs = sorted(SOURCE_QT_IMG.glob("*.png"))[:test_count]
    for ti in test_imgs:
        link_or_copy(ti, img_out / ti.name, use_copy)
        lab = SOURCE_QT_LAB / f"{ti.stem}.txt"
        if lab.is_file():
            link_or_copy(lab, lab_out / lab.name, use_copy)
    cfg.save()


def main() -> int:
    parser = argparse.ArgumentParser(description="准备 regression job")
    parser.add_argument("--display-name", default="回归测试")
    parser.add_argument("--onnx-name", default="wrs_fp16_final2")
    parser.add_argument("--cali-count", type=int, default=496)
    parser.add_argument("--test-count", type=int, default=281)
    parser.add_argument("--copy", action="store_true")
    parser.add_argument("--validate", action="store_true", default=True)
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument("--task-id", default=None, help="固定任务目录名（如 regression_auto）")
    parser.add_argument("--use-shared-datasets", action="store_true", help="使用 shared_data 默认标定/测试集")
    parser.add_argument("--min-cali", type=int, default=400)
    parser.add_argument("--min-test", type=int, default=100)
    args = parser.parse_args()
    if args.no_validate:
        args.validate = False

    cfg = JobConfig.create_task(args.display_name, onnx_name=args.onnx_name, task_id=args.task_id)
    if args.use_shared_datasets:
        from shared_datasets import attach_datasets_to_job, bootstrap_default_datasets  # noqa: E402

        bootstrap_default_datasets()
        cfg.cali_dataset_id = "default_496"
        cfg.test_dataset_id = "moon_earth_257"
        link_or_copy(SOURCE_PT, cfg.input_dir / "model.pt", args.copy)
        attach_datasets_to_job(
            cfg.input_dir,
            cali_dataset_id=cfg.cali_dataset_id,
            test_dataset_id=cfg.test_dataset_id,
        )
    else:
        populate_job(cfg, cali_count=args.cali_count, test_count=args.test_count, use_copy=args.copy)
    cfg.min_cali_images = args.min_cali
    cfg.min_test_images = args.min_test
    cfg.save()
    print(f"Job ready: {cfg.job_root}")

    if args.validate:
        r = subprocess.run(
            [
                sys.executable,
                str(_ROOT / "07_validate_input.py"),
                "--job-dir",
                str(cfg.job_root),
                "--min-cali",
                str(args.min_cali),
                "--min-test",
                str(args.min_test),
            ],
            capture_output=True,
            text=True,
        )
        print(r.stdout)
        if r.returncode != 0:
            print(r.stderr, file=sys.stderr)
            return r.returncode
        from input_manifest import write_input_manifest  # noqa: E402

        write_input_manifest(cfg, validation_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
