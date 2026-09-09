#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FPGA 离线评估：预测 JSON vs labels_1280.json。
逻辑来自 sample_4per_class/eval_fpga_json.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_LIB = Path(__file__).resolve().parent / "_lib"
if str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))


def load_json_dict(path: Path) -> Dict[str, dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for entry in data:
        name = entry.get("image_name")
        if name:
            out[name] = entry
    return out


def box_iou_xywh(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    def to_xyxy(cx, cy, w, h):
        return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

    ax1, ay1, ax2, ay2 = to_xyxy(ax, ay, aw, ah)
    bx1, by1, bx2, by2 = to_xyxy(bx, by, bw, bh)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def kpt_dist(p1, p2) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def match_boxes_iou(gt_boxes, pred_boxes, iou_thresh):
    if not gt_boxes or not pred_boxes:
        return [], list(range(len(gt_boxes))), list(range(len(pred_boxes)))
    ious = []
    for gi, gb in enumerate(gt_boxes):
        for pi, pb in enumerate(pred_boxes):
            giou = box_iou_xywh(
                (gb["x_center_1280"], gb["y_center_1280"], gb["width_1280"], gb["height_1280"]),
                (pb["x_center_1280"], pb["y_center_1280"], pb["width_1280"], pb["height_1280"]),
            )
            if giou >= iou_thresh:
                ious.append((giou, gi, pi))
    ious.sort(reverse=True, key=lambda x: x[0])
    matched_g, matched_p, matches = set(), set(), []
    for _, gi, pi in ious:
        if gi in matched_g or pi in matched_p:
            continue
        matched_g.add(gi)
        matched_p.add(pi)
        matches.append((gi, pi))
    unmatched_g = [i for i in range(len(gt_boxes)) if i not in matched_g]
    unmatched_p = [i for i in range(len(pred_boxes)) if i not in matched_p]
    return matches, unmatched_g, unmatched_p


def run_eval_json(gt_path: Path, pred_path: Path, iou_thresh: float = 0.5) -> dict:
    gt_dict = load_json_dict(gt_path)
    pred_dict = load_json_dict(pred_path)
    all_names = sorted(set(gt_dict.keys()) | set(pred_dict.keys()))
    tp = fp = fn = tn = 0
    total_gt_boxes = total_noobj = 0
    kpt_diffs: List[float] = []

    for name in all_names:
        gt_entry = gt_dict.get(name, {"boxes": [], "keypoints": []})
        pr_entry = pred_dict.get(name, {"boxes": [], "keypoints": []})
        gt_boxes = gt_entry.get("boxes") or []
        pr_boxes = pr_entry.get("boxes") or []
        gt_kpts_all = gt_entry.get("keypoints") or []
        pr_kpts_all = pr_entry.get("keypoints") or []
        total_gt_boxes += len(gt_boxes)

        if not gt_boxes and not pr_boxes:
            tp += 1
            total_noobj += 1
            continue
        if not gt_boxes and pr_boxes:
            fp += len(pr_boxes)
            total_noobj += 1
            continue
        if gt_boxes and not pr_boxes:
            fn += len(gt_boxes)
            continue

        matches, unmatched_g, unmatched_p = match_boxes_iou(gt_boxes, pr_boxes, iou_thresh)
        tp += len(matches)
        fn += len(unmatched_g)
        fp += len(unmatched_p)
        for gi, pi in matches:
            if gi < len(gt_kpts_all) and pi < len(pr_kpts_all):
                gt_k, pr_k = gt_kpts_all[gi], pr_kpts_all[pi]
                if gt_k and pr_k and gt_k[0].get("vis", 0) >= 1:
                    kpt_diffs.append(
                        kpt_dist(
                            (gt_k[0]["x_1280"], gt_k[0]["y_1280"]),
                            (pr_k[0]["x_1280"], pr_k[0]["y_1280"]),
                        )
                    )

    total_cases = total_gt_boxes + total_noobj
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    n_kpt = len(kpt_diffs)
    mean_kpt = sum(kpt_diffs) / n_kpt if n_kpt else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mAP": precision * recall,
        "box_accuracy": (tp + tn) / total_cases if total_cases else 0.0,
        "iou_thresh": iou_thresh,
        "n_images": len(all_names),
        "pixel_error": {
            "mean_px": mean_kpt,
            "max_px": max(kpt_diffs) if kpt_diffs else 0.0,
            "min_px": min(kpt_diffs) if kpt_diffs else 0.0,
            "count": n_kpt,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="FPGA JSON 评估")
    parser.add_argument("--gt", type=Path, default=Path("labels_1280.json"))
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args()
    if not args.gt.is_file():
        print(f"GT 不存在: {args.gt}", file=sys.stderr)
        return 1
    if not args.pred.is_file():
        print(f"预测 JSON 不存在: {args.pred}", file=sys.stderr)
        return 1
    result = run_eval_json(args.gt, args.pred, args.iou)
    out = args.output or args.pred.parent / "fpga_eval_summary.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    pe = result["pixel_error"]
    print(f"precision={result['precision']:.4f} recall={result['recall']:.4f} f1={result['f1']:.4f}")
    print(f"pixel_error mean={pe['mean_px']:.2f}px count={pe['count']}")
    print(f"已写入: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
