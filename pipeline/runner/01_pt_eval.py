#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""步骤 ①：PT 基线评估。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

_LIB = Path(__file__).resolve().parent / "_lib"
if str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))
from bootstrap import bootstrap_platform_script  # noqa: E402

bootstrap_platform_script()

from eval_common import (  # noqa: E402
    EvalRunConfig,
    aggregate_detection_metrics,
    detections_to_pred_boxes,
    evaluate_image_pair,
    list_test_images,
    load_yolo_labels,
    write_eval_outputs,
)
from job_config import JobConfig  # noqa: E402
from preprocess import bgr_for_pt_predict, imread_unicode  # noqa: E402


def run_pt_eval(cfg: JobConfig) -> int:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("错误: 需要 ultralytics", file=sys.stderr)
        return 1

    if not cfg.model_pt.is_file():
        print(f"错误: 未找到模型 {cfg.model_pt}", file=sys.stderr)
        return 1

    ev = EvalRunConfig(
        imgsz=cfg.imgsz,
        nc=cfg.nc,
        conf=cfg.conf,
        iou=cfg.iou,
        max_det=cfg.max_det,
    )
    model = YOLO(str(cfg.model_pt))
    images = list_test_images(cfg.test_images_dir)
    if not images:
        print("错误: 无测试图", file=sys.stderr)
        return 1

    batch_size = max(1, int(os.environ.get("PT_EVAL_BATCH_SIZE", "8")))
    per_image = []
    for start in range(0, len(images), batch_size):
        batch_paths = images[start : start + batch_size]
        prepared = []
        shapes = []
        for img_path in batch_paths:
            img = imread_unicode(img_path)
            orig_h, orig_w = img.shape[:2]
            prepared.append(bgr_for_pt_predict(img, cfg.preprocess_mode, ev.imgsz))
            shapes.append((orig_h, orig_w))
        results = model.predict(
            source=prepared,
            imgsz=ev.imgsz,
            conf=ev.conf,
            iou=ev.iou,
            max_det=ev.max_det,
            batch=batch_size,
            verbose=False,
        )
        for img_path, (orig_h, orig_w), result in zip(batch_paths, shapes, results):
            label_path = cfg.test_labels_dir / f"{img_path.stem}.txt"
            dets = result.boxes
            pred_boxes: list = []
            if dets is not None and len(dets):
                xyxy = dets.xyxy.cpu().numpy()
                confs = dets.conf.cpu().numpy()
                clss = dets.cls.cpu().numpy()
                scale_x = orig_w / ev.imgsz
                scale_y = orig_h / ev.imgsz
                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i]
                    x1 *= scale_x
                    x2 *= scale_x
                    y1 *= scale_y
                    y2 *= scale_y
                    pred_boxes.append(
                        [float(x1), float(y1), float(x2), float(y2), float(confs[i]), float(clss[i])]
                    )

            gt = load_yolo_labels(label_path, orig_w, orig_h, label_wh_ref=cfg.test_label_wh_ref())
            m = evaluate_image_pair(pred_boxes, gt, ev.nc, ev.eval_iou)
            m["image"] = img_path.name
            m["pred_boxes"] = pred_boxes
            m["gt_boxes"] = [list(b) for b in gt]
            per_image.append(m)
        print(f"  PT batch {min(start + batch_size, len(images))}/{len(images)}", flush=True)

    summary = aggregate_detection_metrics(per_image, ev.nc)
    write_eval_outputs(
        cfg.pt_eval_dir(),
        summary,
        per_image,
        "PT Baseline Evaluation",
        test_images_dir=cfg.test_images_dir,
        eval_iou=ev.eval_iou,
    )
    print(f"PT 评估完成: {cfg.pt_eval_dir() / 'summary.json'} | batch={batch_size}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="PT 基线评估")
    parser.add_argument("--job-dir", required=True, type=Path)
    args = parser.parse_args()
    cfg = JobConfig.from_job_dir(args.job_dir)
    return run_pt_eval(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
