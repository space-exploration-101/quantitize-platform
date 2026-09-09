#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""步骤 ④：按用户测试集生成 fpga_test_pack（方案 A png2bin 往返）。"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import sys
from pathlib import Path

_LIB = Path(__file__).resolve().parent / "_lib"
if str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))
from bootstrap import bootstrap_platform_script  # noqa: E402

QUANTITIZE_DIR = bootstrap_platform_script()
from script_registry import EngineScripts, PlatformScripts, engine_script, platform_script  # noqa: E402
from env import run_script  # noqa: E402
from job_config import JobConfig  # noqa: E402

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
FPGA_SIZE = 2000


def _convert_image(item: tuple[str, str, str, int, str]) -> tuple[str, str, str]:
    """Process one image in a worker; outputs are disjoint and deterministic."""
    img_path, bin_path, side_path, target_size, preprocess_mode = item
    import cv2
    from png_bin_converter import bin_to_png, png_to_bin

    cv2.setNumThreads(1)
    # The converter's diagnostic output is useful for the CLI but prohibitively
    # noisy for 840 worker jobs.  The parent prints bounded progress instead.
    with contextlib.redirect_stdout(io.StringIO()):
        png_to_bin(
            img_path,
            bin_path,
            target_size=target_size,
            preprocess_mode=preprocess_mode,
        )
        bin_to_png(
            bin_path,
            side_path,
            width=target_size,
            height=target_size,
            preprocess_mode=preprocess_mode,
        )
    return img_path, bin_path, side_path


def list_test_images(images_dir: Path) -> list[Path]:
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def _clean_generated_files(directory: Path) -> None:
    if not directory.is_dir():
        return
    for path in directory.iterdir():
        if path.is_file():
            path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 FPGA 测试包")
    parser.add_argument("--job-dir", required=True, type=Path)
    args = parser.parse_args()
    cfg = JobConfig.from_job_dir(args.job_dir)
    images = list_test_images(cfg.test_images_dir)
    if len(images) < cfg.min_test_images:
        print(
            f"错误: 测试图 {len(images)} 张，少于要求 {cfg.min_test_images}",
            file=sys.stderr,
        )
        return 1

    pack = cfg.fpga_test_pack_dir
    bins_dir = pack / "bins"
    side_dir = pack / "side_view"
    scripts_dir = pack / "scripts"
    config_dir = pack / "config"
    for d in (bins_dir, side_dir, scripts_dir, config_dir):
        d.mkdir(parents=True, exist_ok=True)
    _clean_generated_files(bins_dir)
    _clean_generated_files(side_dir)

    requested_workers = max(1, int(os.environ.get("FPGA_PACK_WORKERS", "8")))
    workers = min(requested_workers, len(images), os.cpu_count() or 1)
    work = [
        (
            str(img_path),
            str(bins_dir / f"{img_path.stem}.bin"),
            str(side_dir / f"{img_path.stem}.png"),
            FPGA_SIZE,
            cfg.preprocess_mode,
        )
        for img_path in images
    ]
    if workers == 1:
        converted = [_convert_image(item) for item in work]
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as pool:
            converted = list(pool.map(_convert_image, work, chunksize=1))

    manifest_entries = []
    for idx, (img_s, bin_s, side_s) in enumerate(converted):
        img_path = Path(img_s)
        bin_path = Path(bin_s)
        side_path = Path(side_s)
        label_path = cfg.test_labels_dir / f"{img_path.stem}.txt"
        manifest_entries.append(
            {
                "index": idx,
                "image_name": img_path.name,
                "source_image": str(img_path.relative_to(cfg.job_root)),
                "bin": str(bin_path.relative_to(pack)),
                "side_view": str(side_path.relative_to(pack)),
                "label": str(label_path.relative_to(cfg.job_root)) if label_path.is_file() else None,
            }
        )
        if (idx + 1) % 50 == 0 or idx + 1 == len(converted):
            print(f"  converted {idx + 1}/{len(converted)}", flush=True)

    (pack / "manifest.json").write_text(
        json.dumps(manifest_entries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # 标签 JSON
    labels_script = platform_script(PlatformScripts.YOLO_LABELS_TO_JSON)
    r = run_script(labels_script, ["--job-dir", str(cfg.job_root), "-o", str(pack / "labels_1280.json")])
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        return r.returncode

    # 拷贝离线评估脚本（唯一测试脚本）
    eval_script = Path(__file__).resolve().parent / "eval_fpga_results.py"
    shutil.copy2(eval_script, scripts_dir / "eval_fpga_results.py")

    job_meta = {
        "job_id": cfg.job_id,
        "display_name": cfg.display_name,
        "onnx_name": cfg.onnx_name,
        "nc": cfg.nc,
        "imgsz": cfg.imgsz,
        "conf": cfg.conf,
        "iou": cfg.iou,
        "fpga_input_size": FPGA_SIZE,
        "test_image_count": len(images),
        "preprocess_mode": cfg.preprocess_mode,
    }
    (config_dir / "job_meta.json").write_text(
        json.dumps(job_meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (config_dir / "fpga_ddr_layout.json").write_text(
        json.dumps(
            {
                "bytes_per_row": 4096,
                "width": FPGA_SIZE,
                "height": FPGA_SIZE,
                "bit_depth": 12,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    readme = pack / "README.md"
    readme.write_text(
        "# FPGA 测试包\n\n"
        f"- 测试图数量: {len(images)}\n"
        f"- 预处理模式: `{cfg.preprocess_mode}`\n"
        "- 方案 A: 源图 → 2000 png2bin → bin2png 侧视\n"
        "- `passthrough` 模式下：源图按 R-only 处理，side_view 也保存为 R=gray,G=B=0\n"
        "- 评估标签: labels_1280.json\n"
        "- 离线评估:\n\n"
        "```bash\n"
        "python3 scripts/eval_fpga_results.py \\\n"
        "  --gt labels_1280.json \\\n"
        "  --pred /path/to/fpga_results_1280.json\n"
        "```\n",
        encoding="utf-8",
    )
    print(f"fpga_test_pack 已生成: {pack} ({len(images)} 张, workers={workers})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
