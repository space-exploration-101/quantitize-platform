#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""评估公共逻辑：NMS、检测指标、pixel error、报告输出。"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from constants import IMAGE_EXTS


def list_test_images(images_dir: Path) -> List[Path]:
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def _parse_yolo_label_line(
    parts: List[str],
    img_width: int,
    img_height: int,
    *,
    label_wh_ref: Optional[int] = None,
) -> Optional[Tuple[float, float, float, float, int]]:
    """解析 YOLO 检测/ pose 标签行 → (x1,y1,x2,y2,cls)。

    pose 尾缀存在时，按项目 readme：关键点即目标框中心，优先于 bbox 的 cx/cy。
    pose 的 w/h 在 test_data 中按原图边长 label_wh_ref（通常 2048）归一化；
    评测图若为 1280，需用 label_wh_ref 反算像素宽高，不能仅用 img_width。
    """
    if len(parts) < 5:
        return None
    cls = int(float(parts[0]))
    w_n, h_n = float(parts[3]), float(parts[4])
    cx_n, cy_n = float(parts[1]), float(parts[2])
    is_pose = len(parts) >= 8
    if is_pose:
        vis = int(float(parts[7]))
        if vis > 0:
            cx_n, cy_n = float(parts[5]), float(parts[6])
    cx = cx_n * img_width
    cy = cy_n * img_height
    wh_ref = label_wh_ref if label_wh_ref and label_wh_ref > 0 else img_width
    if is_pose and wh_ref != img_width:
        w = w_n * wh_ref
        h = h_n * wh_ref
    else:
        w = w_n * img_width
        h = h_n * img_height
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, cls)


def load_yolo_labels(
    label_path: Path,
    img_width: int,
    img_height: int,
    *,
    label_wh_ref: Optional[int] = None,
) -> List[Tuple[float, float, float, float, int]]:
    boxes: List[Tuple[float, float, float, float, int]] = []
    if not label_path.is_file():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        parsed = _parse_yolo_label_line(parts, img_width, img_height, label_wh_ref=label_wh_ref)
        if parsed is not None:
            boxes.append(parsed)
    return boxes


def box_center(box: Sequence[float]) -> Tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def calculate_iou(box1: Sequence[float], box2: Sequence[float]) -> float:
    x1_min, y1_min, x1_max, y1_max = box1[:4]
    x2_min, y2_min, x2_max, y2_max = box2[:4]
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    inter = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    a1 = (x1_max - x1_min) * (y1_max - y1_min)
    a2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def calculate_metrics(
    pred_boxes: List[Sequence[float]],
    gt_boxes: List[Sequence[float]],
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    if not pred_boxes and not gt_boxes:
        return {"TP": 0, "FP": 0, "FN": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_boxes:
        return {"TP": 0, "FP": 0, "FN": len(gt_boxes), "precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not gt_boxes:
        return {"TP": 0, "FP": len(pred_boxes), "FN": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_array = np.array(pred_boxes, dtype=float)
    gt_array = np.array(gt_boxes, dtype=float)
    matched_gt: set[int] = set()
    tp = fp = 0
    center_errors: List[float] = []

    order = np.argsort(-pred_array[:, 4]) if pred_array.shape[1] > 4 else np.arange(len(pred_array))
    for pred_idx in order:
        pred = pred_array[pred_idx]
        pred_cls = int(pred[5]) if pred.shape[0] > 5 else int(pred[4])
        best_iou = 0.0
        best_gt = -1
        for gi, gt in enumerate(gt_array):
            if gi in matched_gt:
                continue
            gt_cls = int(gt[4])
            if gt_cls != pred_cls:
                continue
            iou = calculate_iou(pred[:4], gt[:4])
            if iou > best_iou:
                best_iou = iou
                best_gt = gi
        if best_iou >= iou_threshold and best_gt >= 0:
            tp += 1
            matched_gt.add(best_gt)
            pc = box_center(pred[:4])
            gc = box_center(gt_array[best_gt][:4])
            center_errors.append(float(np.hypot(pc[0] - gc[0], pc[1] - gc[1])))
        else:
            fp += 1
    fn = len(gt_boxes) - len(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "center_errors_px": center_errors,
    }


def match_detection_assignments(
    pred_boxes: List[Sequence[float]],
    gt_boxes: List[Sequence[float]],
    iou_threshold: float = 0.5,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """返回 (tp 对 pred_idx,gt_idx,iou), fp 预测下标, fn 真值下标。"""
    if not pred_boxes:
        return [], [], list(range(len(gt_boxes)))
    if not gt_boxes:
        return [], list(range(len(pred_boxes))), []

    pred_array = np.array(pred_boxes, dtype=float)
    gt_array = np.array(gt_boxes, dtype=float)
    matched_gt: set[int] = set()
    tp_pairs: List[Tuple[int, int, float]] = []
    fp_indices: List[int] = []

    order = np.argsort(-pred_array[:, 4]) if pred_array.shape[1] > 4 else np.arange(len(pred_array))
    for pred_idx in order:
        pred = pred_array[pred_idx]
        pred_cls = int(pred[5]) if pred.shape[0] > 5 else int(pred[4])
        best_iou = 0.0
        best_gt = -1
        for gi, gt in enumerate(gt_array):
            if gi in matched_gt:
                continue
            if int(gt[4]) != pred_cls:
                continue
            iou = calculate_iou(pred[:4], gt[:4])
            if iou > best_iou:
                best_iou = iou
                best_gt = gi
        if best_iou >= iou_threshold and best_gt >= 0:
            tp_pairs.append((int(pred_idx), int(best_gt), float(best_iou)))
            matched_gt.add(best_gt)
        else:
            fp_indices.append(int(pred_idx))
    fn_indices = [i for i in range(len(gt_boxes)) if i not in matched_gt]
    return tp_pairs, fp_indices, fn_indices


def split_error_assignments(
    pred_boxes: List[Sequence[float]],
    gt_boxes: List[Sequence[float]],
    iou_threshold: float = 0.5,
) -> Tuple[
    List[Tuple[int, int, float]],
    List[int],
    List[int],
    List[Tuple[int, int, float]],
]:
    """同类别匹配后，将高 IoU 但类别不一致的 pred/gt 标为 class error（非 FP/FN）。"""
    tp_pairs, fp_indices, fn_indices = match_detection_assignments(
        pred_boxes, gt_boxes, iou_threshold
    )
    if not fp_indices or not fn_indices:
        return tp_pairs, fp_indices, fn_indices, []

    pred_array = np.array(pred_boxes, dtype=float)
    gt_array = np.array(gt_boxes, dtype=float)
    fp_set = set(fp_indices)
    fn_set = set(fn_indices)
    candidates: List[Tuple[float, int, int]] = []

    for pi in fp_indices:
        pred = pred_array[pi]
        pred_cls = int(pred[5]) if pred.shape[0] > 5 else int(pred[4])
        for gi in fn_indices:
            gt = gt_array[gi]
            if int(gt[4]) == pred_cls:
                continue
            iou = calculate_iou(pred[:4], gt[:4])
            if iou >= iou_threshold:
                candidates.append((iou, pi, gi))

    candidates.sort(key=lambda x: -x[0])
    class_error_pairs: List[Tuple[int, int, float]] = []
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    for iou, pi, gi in candidates:
        if pi not in fp_set or gi not in fn_set:
            continue
        if pi in used_pred or gi in used_gt:
            continue
        class_error_pairs.append((pi, gi, float(iou)))
        used_pred.add(pi)
        used_gt.add(gi)
        fp_set.discard(pi)
        fn_set.discard(gi)

    return tp_pairs, sorted(fp_set), sorted(fn_set), class_error_pairs


def _box_to_json_gt(box: Sequence[float]) -> Dict[str, Any]:
    return {
        "x1": float(box[0]),
        "y1": float(box[1]),
        "x2": float(box[2]),
        "y2": float(box[3]),
        "cls": int(box[4]),
    }


def _box_to_json_pred(box: Sequence[float]) -> Dict[str, Any]:
    if len(box) > 5:
        return {
            "x1": float(box[0]),
            "y1": float(box[1]),
            "x2": float(box[2]),
            "y2": float(box[3]),
            "conf": float(box[4]),
            "cls": int(box[5]),
        }
    return {
        "x1": float(box[0]),
        "y1": float(box[1]),
        "x2": float(box[2]),
        "y2": float(box[3]),
        "conf": 1.0,
        "cls": int(box[4]),
    }


def draw_error_case_overlay(
    img_bgr: np.ndarray,
    gt_boxes: List[Sequence[float]],
    pred_boxes: List[Sequence[float]],
    tp_pairs: List[Tuple[int, int, float]],
    fp_indices: List[int],
    fn_indices: List[int],
    class_error_pairs: Optional[List[Tuple[int, int, float]]] = None,
) -> np.ndarray:
    """真值绿框、TP 蓝框、真 FP 红框、真 FN 黄描边、类别错 magenta。"""
    vis = img_bgr.copy()
    tp_pred = {p for p, _, _ in tp_pairs}
    class_error_pairs = class_error_pairs or []
    ce_pred = {p for p, _, _ in class_error_pairs}
    ce_gt = {g for _, g, _ in class_error_pairs}
    fn_set = set(fn_indices) - ce_gt

    for gi, gt in enumerate(gt_boxes):
        x1, y1, x2, y2 = (int(gt[0]), int(gt[1]), int(gt[2]), int(gt[3]))
        cls = int(gt[4])
        if gi in ce_gt:
            color = (255, 0, 255)
            tag = f"ClassErr GT{cls}"
        else:
            color = (0, 200, 0)
            tag = f"FN GT{cls}" if gi in fn_set else f"GT{cls}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        if gi in fn_set:
            cv2.rectangle(vis, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 255, 255), 1)
        cv2.putText(vis, tag, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    for pi, pred in enumerate(pred_boxes):
        x1, y1, x2, y2 = (int(pred[0]), int(pred[1]), int(pred[2]), int(pred[3]))
        if len(pred) > 5:
            conf, cls = float(pred[4]), int(pred[5])
            tag = f"P{cls} {conf:.2f}"
        else:
            cls, tag = int(pred[4]), f"P{int(pred[4])}"
        if pi in tp_pred:
            color = (255, 160, 0)
            tag = "TP " + tag
        elif pi in ce_pred:
            color = (255, 0, 255)
            tag = "ClassErr " + tag
        else:
            color = (0, 0, 255)
            tag = "FP " + tag
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, tag, (x1, min(vis.shape[0] - 4, y2 + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    cv2.putText(
        vis,
        "Green=GT Blue=TP Red=FP Yellow outline=FN Magenta=ClassErr",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return vis


def _load_image_for_overlay(
    img_path: Path,
    *,
    overlay_imgsz: Optional[int] = None,
) -> np.ndarray:
    """读取 overlay 底图；若指定 overlay_imgsz，则 resize 到与框坐标同一空间。"""
    from preprocess import imread_unicode

    img = imread_unicode(img_path)
    if overlay_imgsz is not None and overlay_imgsz > 0:
        h, w = img.shape[:2]
        if h != overlay_imgsz or w != overlay_imgsz:
            img = cv2.resize(img, (overlay_imgsz, overlay_imgsz), interpolation=cv2.INTER_LINEAR)
    return img


def export_eval_error_cases(
    out_dir: Path,
    records: List[Dict[str, Any]],
    test_images_dir: Path,
    *,
    eval_iou: float = 0.5,
    max_cases: int = 80,
    overlay_imgsz: Optional[int] = None,
) -> int:
    """为有问题的图生成 overlay 与 error_cases.json（供 Web 展示）。"""

    error_dir = out_dir / "error_cases"
    if error_dir.exists():
        for p in error_dir.iterdir():
            if p.is_file():
                p.unlink()
    else:
        error_dir.mkdir(parents=True, exist_ok=True)

    cases: List[Dict[str, Any]] = []
    pending: List[Tuple[Any, ...]] = []

    for rec in records:
        img_name = rec["image"]
        img_path = test_images_dir / img_name
        if not img_path.is_file():
            continue
        pred_boxes = rec.get("pred_boxes") or []
        gt_boxes = rec.get("gt_boxes") or []
        gt_for_match = [(*b[:4], b[4]) for b in gt_boxes]
        tp_pairs, fp_idx, fn_idx, ce_pairs = split_error_assignments(
            pred_boxes, gt_for_match, eval_iou
        )
        if not fp_idx and not fn_idx and not ce_pairs:
            continue
        pending.append(
            (img_path, img_name, pred_boxes, gt_boxes, tp_pairs, fp_idx, fn_idx, ce_pairs, rec)
        )

    pending.sort(
        key=lambda x: (len(x[7]), len(x[5]) + len(x[6])),
        reverse=True,
    )

    max_per_group = max(10, (max_cases + 2) // 3)
    selected: Dict[str, List[Tuple[Any, ...]]] = {
        "class_error": [],
        "fp": [],
        "fn": [],
    }
    seen_images: set[str] = set()
    for item in pending:
        ce, fp, fn = len(item[7]), len(item[5]), len(item[6])
        img_name = item[1]
        if ce > 0 and len(selected["class_error"]) < max_per_group:
            selected["class_error"].append(item)
            seen_images.add(img_name)
        if fp > 0 and len(selected["fp"]) < max_per_group:
            selected["fp"].append(item)
            seen_images.add(img_name)
        if fn > 0 and len(selected["fn"]) < max_per_group:
            selected["fn"].append(item)
            seen_images.add(img_name)

    # 按 image 去重后生成 overlay（同一图只渲染一次）
    unique_items: Dict[str, Tuple[Any, ...]] = {}
    for bucket in selected.values():
        for item in bucket:
            unique_items.setdefault(item[1], item)

    groups: Dict[str, List[Dict[str, Any]]] = {
        "class_error": [],
        "fp": [],
        "fn": [],
    }
    cases: List[Dict[str, Any]] = []
    case_by_image: Dict[str, Dict[str, Any]] = {}

    for img_path, img_name, pred_boxes, gt_boxes, tp_pairs, fp_idx, fn_idx, ce_pairs, rec in unique_items.values():
        overlay_name = ""
        try:
            img = _load_image_for_overlay(img_path, overlay_imgsz=overlay_imgsz)
            overlay = draw_error_case_overlay(
                img, gt_boxes, pred_boxes, tp_pairs, fp_idx, fn_idx, ce_pairs
            )
            overlay_name = f"{Path(img_name).stem}.jpg"
            cv2.imwrite(str(error_dir / overlay_name), overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        except Exception as exc:
            print(f"警告: 无法生成 error overlay {img_name}: {exc}", file=sys.stderr)
        case = {
            "image": img_name,
            "overlay": overlay_name,
            "TP": rec.get("TP", 0),
            "class_error": len(ce_pairs),
            "FP": len(fp_idx),
            "FN": len(fn_idx),
            "gt_boxes": [_box_to_json_gt(b) for b in gt_boxes],
            "pred_boxes": [_box_to_json_pred(b) for b in pred_boxes],
            "tp_pairs": [{"pred": p, "gt": g, "iou": round(i, 4)} for p, g, i in tp_pairs],
            "class_error_pairs": [
                {"pred": p, "gt": g, "iou": round(i, 4)} for p, g, i in ce_pairs
            ],
            "fp_pred_indices": fp_idx,
            "fn_gt_indices": fn_idx,
        }
        cases.append(case)
        case_by_image[img_name] = case

    for bucket, items in selected.items():
        for item in items:
            case = case_by_image.get(item[1])
            if case:
                groups[bucket].append(case)

    (out_dir / "error_cases.json").write_text(
        json.dumps(
            {
                "count": len(cases),
                "groups": groups,
                "group_counts": {k: len(v) for k, v in groups.items()},
                "cases": cases,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return len(unique_items)


def export_eval_success_cases(
    out_dir: Path,
    records: List[Dict[str, Any]],
    test_images_dir: Path,
    *,
    eval_iou: float = 0.5,
    max_cases: int = 10,
    overlay_imgsz: Optional[int] = None,
) -> int:
    """全检对（TP>0 且 FP=FN=0）抽样 overlay，供 Web metrics 展示。"""
    import random

    success_dir = out_dir / "success_cases"
    if success_dir.exists():
        for p in success_dir.iterdir():
            if p.is_file():
                p.unlink()
    else:
        success_dir.mkdir(parents=True, exist_ok=True)

    perfect: List[Tuple[Any, ...]] = []
    fallback: List[Tuple[Any, ...]] = []

    for rec in records:
        if rec.get("TP", 0) <= 0:
            continue
        img_name = rec["image"]
        img_path = test_images_dir / img_name
        if not img_path.is_file():
            continue
        pred_boxes = rec.get("pred_boxes") or []
        gt_boxes = rec.get("gt_boxes") or []
        gt_for_match = [(*b[:4], b[4]) for b in gt_boxes]
        tp_pairs, fp_idx, fn_idx = match_detection_assignments(pred_boxes, gt_for_match, eval_iou)
        item = (img_path, img_name, pred_boxes, gt_boxes, tp_pairs, fp_idx, fn_idx, rec)
        if rec.get("FP", 0) == 0 and rec.get("FN", 0) == 0:
            perfect.append(item)
        else:
            fallback.append(item)

    pool = perfect if len(perfect) >= max_cases else perfect + sorted(
        fallback, key=lambda x: (x[7].get("FP", 0) + x[7].get("FN", 0), -x[7].get("TP", 0))
    )
    rng = random.Random(42)
    rng.shuffle(pool)
    pool = pool[:max_cases]

    cases: List[Dict[str, Any]] = []
    for img_path, img_name, pred_boxes, gt_boxes, tp_pairs, fp_idx, fn_idx, rec in pool:
        overlay_name = ""
        try:
            img = _load_image_for_overlay(img_path, overlay_imgsz=overlay_imgsz)
            overlay = draw_error_case_overlay(img, gt_boxes, pred_boxes, tp_pairs, fp_idx, fn_idx)
            overlay_name = f"{Path(img_name).stem}.jpg"
            cv2.imwrite(str(success_dir / overlay_name), overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        except Exception as exc:
            print(f"警告: 无法生成 success overlay {img_name}: {exc}", file=sys.stderr)
        cases.append(
            {
                "image": img_name,
                "overlay": overlay_name,
                "TP": rec.get("TP", 0),
                "FP": rec.get("FP", 0),
                "FN": rec.get("FN", 0),
            }
        )

    (out_dir / "success_cases.json").write_text(
        json.dumps({"count": len(cases), "cases": cases}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return len(cases)


def calculate_class_metrics(
    pred_boxes: List[Sequence[float]],
    gt_boxes: List[Sequence[float]],
    num_classes: int,
    iou_threshold: float = 0.5,
) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for cls_id in range(num_classes):
        pred_cls = [b for b in pred_boxes if int(b[5] if len(b) > 5 else b[4]) == cls_id]
        gt_cls = [b for b in gt_boxes if int(b[4]) == cls_id]
        out[cls_id] = calculate_metrics(pred_cls, gt_cls, iou_threshold)
    return out


def run_python_nms(
    prediction: np.ndarray,
    conf_thres: float = 0.25,
    iou_thres: float = 0.7,
    max_det: int = 300,
    nc: int = 6,
) -> np.ndarray:
    batch = run_python_nms_batch(prediction, conf_thres, iou_thres, max_det, nc)
    return batch[0] if batch else np.array([])


def ensure_nms_imported() -> None:
    """在创建 ORT CUDA Session 之前导入 NMS（避免 cuDNN 路径干扰 torch）。"""
    from std_platform import pin_stdlib_platform

    pin_stdlib_platform()
    from ultralytics.utils.nms import non_max_suppression  # noqa: F401, WPS433


def run_python_nms_batch(
    prediction: np.ndarray,
    conf_thres: float = 0.25,
    iou_thres: float = 0.7,
    max_det: int = 300,
    nc: int = 6,
) -> List[np.ndarray]:
    """对 batch 模型输出逐图 NMS，返回长度为 B 的检测列表。"""
    ensure_nms_imported()
    from ultralytics.utils.nms import non_max_suppression
    import torch

    if prediction.ndim == 2:
        prediction = np.expand_dims(prediction, axis=0)
    pred_t = torch.from_numpy(prediction.astype(np.float32)).cpu()
    outputs = non_max_suppression(pred_t, conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det, nc=nc)
    out: List[np.ndarray] = []
    for det in outputs:
        if det is not None and len(det):
            out.append(det.numpy())
        else:
            out.append(np.array([]))
    return out


def create_onnx_eval_session(model_path: Path):
    """创建仅推理检测头的 ORT Session（优先 CUDA）。"""
    from env import apply_gpu_runtime_env

    apply_gpu_runtime_env()
    import onnxruntime as ort

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(model_path), providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_shape = session.get_inputs()[0].shape
    max_batch = 1
    if input_shape and isinstance(input_shape[0], int) and input_shape[0] > 0:
        max_batch = int(input_shape[0])
    print(f"ONNX 评估模型: {model_path.name}")
    print(f"ORT providers: {session.get_providers()}")
    print(f"输入 shape: {input_shape} (max_batch={max_batch})")
    return session, input_name, output_name, max_batch


def run_onnx_batch(
    session: Any,
    input_name: str,
    output_name: str,
    batch_tensor: np.ndarray,
    conf_thres: float,
    iou_thres: float,
    max_det: int,
    nc: int,
    *,
    max_ort_batch: int = 1,
) -> List[np.ndarray]:
    if batch_tensor.ndim == 3:
        batch_tensor = np.expand_dims(batch_tensor, axis=0)
    b = batch_tensor.shape[0]
    preds: List[np.ndarray] = []
    chunk = max(1, max_ort_batch)
    for start in range(0, b, chunk):
        sub = batch_tensor[start : start + chunk]
        if sub.shape[0] == 1 or chunk == 1:
            for i in range(sub.shape[0]):
                one = sub[i : i + 1]
                pred = session.run([output_name], {input_name: one})[0]
                preds.append(pred)
        else:
            pred = session.run([output_name], {input_name: sub})[0]
            for i in range(pred.shape[0]):
                preds.append(pred[i : i + 1])
    prediction = np.concatenate(preds, axis=0)
    if prediction.dtype != np.float32:
        prediction = prediction.astype(np.float32)
    return run_python_nms_batch(prediction, conf_thres, iou_thres, max_det, nc)


def scale_boxes_to_orig(
    boxes: np.ndarray,
    prep_shape: Tuple[int, int],
    orig_shape: Tuple[int, int],
) -> np.ndarray:
    if len(boxes) == 0:
        return boxes
    orig_h, orig_w = orig_shape
    prep_h, prep_w = prep_shape
    gain = min(prep_h / orig_h, prep_w / orig_w)
    pad_x = round((prep_w - orig_w * gain) / 2 - 0.1)
    pad_y = round((prep_h - orig_h * gain) / 2 - 0.1)
    out = boxes.copy()
    out[:, 0] -= pad_x
    out[:, 2] -= pad_x
    out[:, 1] -= pad_y
    out[:, 3] -= pad_y
    out[:, :4] /= gain
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, orig_w)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, orig_h)
    return out


def aggregate_pixel_error(all_center_errors: List[float]) -> Dict[str, float]:
    if not all_center_errors:
        return {"mean_px": 0.0, "max_px": 0.0, "min_px": 0.0, "count": 0}
    arr = np.array(all_center_errors, dtype=float)
    return {
        "mean_px": float(np.mean(arr)),
        "max_px": float(np.max(arr)),
        "min_px": float(np.min(arr)),
        "count": int(len(arr)),
    }


def aggregate_detection_metrics(
    per_image: List[Dict[str, Any]],
    num_classes: int,
) -> Dict[str, Any]:
    tp = sum(m["TP"] for m in per_image)
    fp = sum(m["FP"] for m in per_image)
    fn = sum(m["FN"] for m in per_image)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    per_class: Dict[str, Dict[str, float]] = {}
    aps: List[float] = []
    for cls_id in range(num_classes):
        c_tp = sum(m.get("per_class", {}).get(cls_id, {}).get("TP", 0) for m in per_image)
        c_fp = sum(m.get("per_class", {}).get(cls_id, {}).get("FP", 0) for m in per_image)
        c_fn = sum(m.get("per_class", {}).get(cls_id, {}).get("FN", 0) for m in per_image)
        c_p = c_tp / (c_tp + c_fp) if (c_tp + c_fp) else 0.0
        c_r = c_tp / (c_tp + c_fn) if (c_tp + c_fn) else 0.0
        c_f1 = 2 * c_p * c_r / (c_p + c_r) if (c_p + c_r) else 0.0
        per_class[str(cls_id)] = {"precision": c_p, "recall": c_r, "f1": c_f1, "ap": c_p * c_r}
        aps.append(c_p * c_r)

    all_err = [e for m in per_image for e in m.get("center_errors_px", [])]
    pixel = aggregate_pixel_error(all_err)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mAP": float(np.mean(aps)) if aps else 0.0,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "pixel_error": pixel,
        "per_class": per_class,
        "n_images": len(per_image),
    }


def detections_to_pred_boxes(detections: np.ndarray) -> List[List[float]]:
    if detections is None or len(detections) == 0:
        return []
    return detections.tolist()


def run_onnx_on_image(
    session: Any,
    input_name: str,
    output_name: str,
    img_tensor: np.ndarray,
    conf_thres: float,
    iou_thres: float,
    max_det: int,
    nc: int,
) -> np.ndarray:
    dets = run_onnx_batch(
        session, input_name, output_name, img_tensor, conf_thres, iou_thres, max_det, nc
    )
    return dets[0] if dets else np.array([])


@dataclass
class EvalRunConfig:
    imgsz: int = 1280
    nc: int = 6
    conf: float = 0.25
    iou: float = 0.7
    max_det: int = 300
    eval_iou: float = 0.5


def evaluate_image_pair(
    pred_boxes: List[List[float]],
    gt_boxes: List[Tuple[float, float, float, float, int]],
    nc: int,
    eval_iou: float = 0.5,
) -> Dict[str, Any]:
    gt_for_metrics = [(*b[:4], b[4]) for b in gt_boxes]
    metrics = calculate_metrics(pred_boxes, gt_for_metrics, eval_iou)
    per_class = calculate_class_metrics(pred_boxes, gt_for_metrics, nc, eval_iou)
    return {
        **{k: metrics[k] for k in ("TP", "FP", "FN", "precision", "recall", "f1")},
        "center_errors_px": metrics.get("center_errors_px", []),
        "per_class": per_class,
    }


def write_eval_outputs(
    out_dir: Path,
    summary: Dict[str, Any],
    per_image: List[Dict[str, Any]],
    title: str,
    *,
    test_images_dir: Optional[Path] = None,
    eval_iou: float = 0.5,
    overlay_imgsz: Optional[int] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    slim = []
    for row in per_image:
        slim.append({k: v for k, v in row.items() if k not in ("pred_boxes", "gt_boxes")})
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "per_image.json").write_text(json.dumps(slim, indent=2, ensure_ascii=False), encoding="utf-8")
    if test_images_dir is not None:
        n_err = export_eval_error_cases(
            out_dir,
            per_image,
            test_images_dir,
            eval_iou=eval_iou,
            overlay_imgsz=overlay_imgsz,
        )
        n_ok = export_eval_success_cases(
            out_dir,
            per_image,
            test_images_dir,
            eval_iou=eval_iou,
            overlay_imgsz=overlay_imgsz,
        )
        summary = dict(summary)
        summary["error_case_count"] = n_err
        summary["success_case_count"] = n_ok
        err_json = out_dir / "error_cases.json"
        if err_json.is_file():
            err_data = json.loads(err_json.read_text(encoding="utf-8"))
            summary["error_group_counts"] = err_data.get("group_counts", {})
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    pe = summary.get("pixel_error", {})
    lines = [
        f"# {title}",
        "",
        "## Detection",
        f"- precision: {summary.get('precision', 0):.4f}",
        f"- recall: {summary.get('recall', 0):.4f}",
        f"- f1: {summary.get('f1', 0):.4f}",
        f"- mAP (macro): {summary.get('mAP', 0):.4f}",
        "",
        "## Pixel Error (matched centers)",
        f"- mean: {pe.get('mean_px', 0):.2f} px",
        f"- max: {pe.get('max_px', 0):.2f} px",
        f"- min: {pe.get('min_px', 0):.2f} px",
        f"- count: {pe.get('count', 0)}",
        "",
    ]
    (out_dir / "evaluation_report.txt").write_text("\n".join(lines), encoding="utf-8")
