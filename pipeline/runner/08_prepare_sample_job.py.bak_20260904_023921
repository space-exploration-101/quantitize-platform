#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从仓库现有数据组装示例 job（供本地/Web 联调）。

数据源（只读，来自 QUANTITIZE_LEGACY_DIR，默认旧 quantitize/）：
  - best.pt
  - cali_data/          （496 张，2048）
  - Quantitative test/quantize_test/  （281 对图+标签，4096，6 类）

默认输出：data/output_data/_sample_demo_job/
默认使用符号链接以节省磁盘；--copy 可改为物理拷贝。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

_LIB = Path(__file__).resolve().parent / "_lib"
if str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))
from bootstrap import bootstrap_platform_script  # noqa: E402
from script_registry import OUTPUT_DATA_ROOT, QUANTITIZE_LEGACY_DIR  # noqa: E402

QUANTITIZE_DIR = bootstrap_platform_script()

SOURCE_PT = QUANTITIZE_LEGACY_DIR / "best.pt"
SOURCE_CALI = QUANTITIZE_LEGACY_DIR / "cali_data"
SOURCE_QT = QUANTITIZE_LEGACY_DIR / "Quantitative test" / "quantize_test"
SOURCE_QT_IMG = SOURCE_QT / "images" / "test"
SOURCE_QT_LAB = SOURCE_QT / "labels" / "test"

DEFAULT_OUT = OUTPUT_DATA_ROOT / "_sample_demo_job"


def _link_or_copy(src: Path, dst: Path, use_copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if use_copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def pick_test_files_per_class(limit: int) -> list[Path]:
    """按类别均衡选取测试图（文件名前缀 aus/lka/...）。"""
    prefixes = ["aus", "lka", "shmeo", "taihu", "vict", "moon"]
    by_prefix: dict[str, list[Path]] = {p: [] for p in prefixes}
    for img in sorted(SOURCE_QT_IMG.glob("*.png")):
        for p in prefixes:
            if img.stem.startswith(p + "_") or img.stem == p:
                by_prefix[p].append(img)
                break
    per = max(1, limit // len(prefixes))
    chosen: list[Path] = []
    for p in prefixes:
        chosen.extend(by_prefix[p][:per])
    # 补足到 limit
    if len(chosen) < limit:
        rest = [p for p in sorted(SOURCE_QT_IMG.glob("*.png")) if p not in chosen]
        chosen.extend(rest[: limit - len(chosen)])
    return sorted(chosen)[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(description="组装示例上传 job")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cali-count", type=int, default=496, help="现网 cali_data 全量")
    parser.add_argument("--test-count", type=int, default=281, help="现网 quantize_test 全量")
    parser.add_argument("--copy", action="store_true", help="物理拷贝而非符号链接")
    parser.add_argument("--validate", action="store_true", help="完成后运行 07_validate_input")
    args = parser.parse_args()

    out = args.output.resolve()
    if not SOURCE_PT.is_file():
        print(f"错误: 缺少 {SOURCE_PT}", file=sys.stderr)
        return 1
    if not SOURCE_CALI.is_dir():
        print(f"错误: 缺少 {SOURCE_CALI}", file=sys.stderr)
        return 1
    if not SOURCE_QT_IMG.is_dir():
        print(f"错误: 缺少 {SOURCE_QT_IMG}", file=sys.stderr)
        return 1

    inp = out / "input"
    cali_out = inp / "cali"
    img_out = inp / "test" / "images"
    lab_out = inp / "test" / "labels"
    for d in (cali_out, img_out, lab_out):
        d.mkdir(parents=True, exist_ok=True)

    _link_or_copy(SOURCE_PT, inp / "model.pt", args.copy)

    cali_files = sorted(SOURCE_CALI.glob("*.png"))[: args.cali_count]
    for cf in cali_files:
        _link_or_copy(cf, cali_out / cf.name, args.copy)

    test_imgs = pick_test_files_per_class(args.test_count)
    for ti in test_imgs:
        _link_or_copy(ti, img_out / ti.name, args.copy)
        lab = SOURCE_QT_LAB / f"{ti.stem}.txt"
        if lab.is_file():
            _link_or_copy(lab, lab_out / lab.name, args.copy)

    job_cfg = {
        "job_id": out.name,
        "onnx_name": "demo",
        "nc": 6,
        "imgsz": 1280,
        "conf": 0.25,
        "iou": 0.7,
        "min_test_images": 100,
        "min_cali_images": 400,
    }
    (out / "job_config.json").write_text(
        json.dumps(job_cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    readme = out / "README.txt"
    readme.write_text(
        "示例 job，由 08_prepare_sample_job.py 生成。\n"
        f"  model.pt ← {SOURCE_PT.name}\n"
        f"  cali/    ← {len(cali_files)} 张 from cali_data/\n"
        f"  test/    ← {len(test_imgs)} 对 from quantize_test/\n"
        "运行检校: python pipeline/runner/07_validate_input.py --job-dir 此目录\n",
        encoding="utf-8",
    )
    print(f"已生成示例 job: {out}")
    print(f"  标定 {len(cali_files)} 张, 测试 {len(test_imgs)} 对")

    if args.validate:
        from validate_input import save_report, validate_job_input  # noqa: WPS433

        report = validate_job_input(out, min_cali=400, min_test=100, nc=6)
        save_report(report, out / "input_validation.json")
        return 0 if report.ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
