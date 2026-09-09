#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""步骤 ③ / ⑤：ONNX 评估（--input-mode direct | fpga_side_view）。"""

from __future__ import annotations

import argparse
import sys
import time
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
    create_onnx_eval_session,
    ensure_nms_imported,
    evaluate_image_pair,
    list_test_images,
    load_yolo_labels,
    run_onnx_batch,
    scale_boxes_to_orig,
    write_eval_outputs,
)
from job_config import JobConfig  # noqa: E402
from preprocess import (  # noqa: E402
    grayscale_to_chw_tensor,
    imread_unicode,
    preprocess_bgr_by_mode,
    to_grayscale_hw,
)

DEFAULT_BATCH_SIZE = 8


def preprocess_chw(
    img_path: Path,
    imgsz: int,
    side_view_mode: bool,
    preprocess_mode: str,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    """返回 (3,H,W) float16 与原始/预处理尺寸。"""
    img = imread_unicode(img_path)
    orig_h, orig_w = img.shape[:2]
    if side_view_mode:
        gray = to_grayscale_hw(img)
        gray = cv2.resize(gray, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        chw = grayscale_to_chw_tensor(gray, (imgsz, imgsz), dtype=np.float16)[0]
    else:
        chw = preprocess_bgr_by_mode(img, preprocess_mode, (imgsz, imgsz), dtype=np.float16)[0]
    return chw, (orig_h, orig_w), (imgsz, imgsz)


def load_eval_batch(
    image_paths: list[Path],
    imgsz: int,
    side_view_mode: bool,
    preprocess_mode: str,
    *,
    workers: int = 4,
) -> tuple[np.ndarray, list[tuple[Path, tuple[int, int], tuple[int, int]]]]:
    from concurrent.futures import ThreadPoolExecutor

    def _one(img_path: Path) -> tuple[np.ndarray, Path, tuple[int, int], tuple[int, int]]:
        chw, orig_shape, prep_shape = preprocess_chw(img_path, imgsz, side_view_mode, preprocess_mode)
        return chw, img_path, orig_shape, prep_shape

    if workers > 1 and len(image_paths) > 1:
        with ThreadPoolExecutor(max_workers=min(workers, len(image_paths))) as pool:
            rows = list(pool.map(_one, image_paths))
    else:
        rows = [_one(p) for p in image_paths]

    chw_list = [r[0] for r in rows]
    meta = [(r[1], r[2], r[3]) for r in rows]
    batch = np.stack(chw_list, axis=0)
    return batch, meta


def resolve_eval_model(cfg: JobConfig, model_path: Path | None) -> Path:
    if model_path is not None:
        return model_path
    quantized = cfg.quantized_onnx()
    if quantized.is_file():
        return quantized
    raise FileNotFoundError(f"未找到量化 ONNX: {quantized}")


def run_onnx_eval(
    cfg: JobConfig,
    model_path: Path | None,
    input_mode: str,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    model_path = resolve_eval_model(cfg, model_path)
    if not model_path.is_file():
        print(f"错误: 模型不存在 {model_path}", file=sys.stderr)
        return 1

    ev = EvalRunConfig(
        imgsz=cfg.imgsz,
        nc=cfg.nc,
        conf=cfg.conf,
        iou=cfg.iou,
        max_det=cfg.max_det,
    )
    side_view = input_mode == "fpga_side_view"
    if side_view:
        images_root = cfg.fpga_test_pack_dir / "side_view"
        title = "ONNX FPGA Side-View Evaluation"
        out_dir = cfg.fpga_eval_dir()
    else:
        images_root = cfg.test_images_dir
        title = "ONNX Direct Evaluation"
        out_dir = cfg.onnx_eval_dir()

    images = list_test_images(images_root)
    if not images:
        print(f"错误: 无测试图 {images_root}", file=sys.stderr)
        return 1

    batch_size = max(1, batch_size)
    ensure_nms_imported()
    session, input_name, output_name, max_ort_batch = create_onnx_eval_session(model_path)

    # CUDA 预热，避免首张图统计失真
    warmup = np.zeros((1, 3, ev.imgsz, ev.imgsz), dtype=np.float16)
    session.run([output_name], {input_name: warmup})

    per_image: list[dict] = []
    t0 = time.perf_counter()
    n_batches = (len(images) + batch_size - 1) // batch_size
    for bi in range(n_batches):
        batch_paths = images[bi * batch_size : (bi + 1) * batch_size]
        batch_tensor, meta = load_eval_batch(
            batch_paths, ev.imgsz, side_view, cfg.preprocess_mode
        )
        detections_list = run_onnx_batch(
            session,
            input_name,
            output_name,
            batch_tensor,
            ev.conf,
            ev.iou,
            ev.max_det,
            ev.nc,
            max_ort_batch=max_ort_batch,
        )
        for (img_path, orig_shape, prep_shape), dets in zip(meta, detections_list):
            # 侧视图为 2000×2000，标签仍对应 1280 源图；指标与 overlay 统一在 imgsz 空间
            eval_shape = (ev.imgsz, ev.imgsz) if side_view else orig_shape
            if len(dets):
                dets = dets.copy()
                dets[:, :4] = scale_boxes_to_orig(dets[:, :4], prep_shape, eval_shape)
            pred_boxes = dets.tolist() if len(dets) else []
            label_path = cfg.test_labels_dir / f"{img_path.stem}.txt"
            gt = load_yolo_labels(
                label_path, eval_shape[1], eval_shape[0], label_wh_ref=cfg.test_label_wh_ref()
            )
            m = evaluate_image_pair(pred_boxes, gt, ev.nc, ev.eval_iou)
            m["image"] = img_path.name
            m["pred_boxes"] = pred_boxes
            m["gt_boxes"] = [list(b) for b in gt]
            per_image.append(m)
        print(
            f"  batch {bi + 1}/{n_batches} ({len(batch_paths)} 张)",
            flush=True,
        )

    elapsed = time.perf_counter() - t0
    summary = aggregate_detection_metrics(per_image, ev.nc)
    # 侧视评测框在 imgsz 空间；overlay 必须用同尺寸底图，不能直接画在原分辨率测试图上
    overlay_images = cfg.test_images_dir if side_view else images_root
    write_eval_outputs(
        out_dir,
        summary,
        per_image,
        title,
        test_images_dir=overlay_images,
        eval_iou=ev.eval_iou,
        overlay_imgsz=ev.imgsz if side_view else None,
    )
    print(
        f"ONNX 评估完成 ({input_mode}): {out_dir / 'summary.json'} "
        f"| {len(images)} 张, batch={batch_size}, {elapsed:.1f}s "
        f"({elapsed / len(images):.3f}s/张)"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="ONNX 多图评估")
    parser.add_argument("--job-dir", required=True, type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="量化 ONNX；默认 {workspace}/{onnx_name}.onnx",
    )
    parser.add_argument(
        "--input-mode",
        choices=("direct", "fpga_side_view"),
        default="direct",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"推理 batch 大小（默认 {DEFAULT_BATCH_SIZE}）",
    )
    args = parser.parse_args()
    cfg = JobConfig.from_job_dir(args.job_dir)
    return run_onnx_eval(
        cfg,
        args.model,
        args.input_mode,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    raise SystemExit(main())
