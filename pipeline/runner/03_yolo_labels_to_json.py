#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO txt 标签 → labels_1280.json（供 FPGA eval 与 pack 使用）。"""

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

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def yolo_txt_to_boxes(txt_path: Path, img_w: int, img_h: int, *, label_wh_ref: int | None = None) -> list:
    from eval_common import _parse_yolo_label_line  # noqa: WPS433

    boxes = []
    if not txt_path.is_file():
        return boxes
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        parsed = _parse_yolo_label_line(parts, img_w, img_h, label_wh_ref=label_wh_ref)
        if parsed is None:
            continue
        x1, y1, x2, y2, cls_id = parsed
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        boxes.append(
            {
                "class_id": cls_id,
                "x_center_1280": cx,
                "y_center_1280": cy,
                "width_1280": x2 - x1,
                "height_1280": y2 - y1,
            }
        )
    return boxes


def build_labels_json(cfg: JobConfig, imgsz: int) -> list:
    label_wh_ref = cfg.test_label_wh_ref()
    entries = []
    for img_path in sorted(cfg.test_images_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        label_path = cfg.test_labels_dir / f"{img_path.stem}.txt"
        entries.append(
            {
                "image_name": img_path.name,
                "boxes": yolo_txt_to_boxes(label_path, imgsz, imgsz, label_wh_ref=label_wh_ref),
                "keypoints": [],
            }
        )
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description="YOLO 标签转 labels_1280.json")
    parser.add_argument("--job-dir", required=True, type=Path)
    parser.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args()
    cfg = JobConfig.from_job_dir(args.job_dir)
    out = args.output or (cfg.fpga_test_pack_dir / "labels_1280.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    data = build_labels_json(cfg, cfg.imgsz)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"已写入 {len(data)} 条: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
