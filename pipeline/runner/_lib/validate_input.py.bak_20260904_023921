#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""上传数据检校（对应 WEB_DESIGN.md §11）。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# 与现有 quantize_test / multi_label_test 一致
DEFAULT_CLASS_NAMES = {
    0: "aus",
    1: "lka",
    2: "shmeo",
    3: "taihu",
    4: "vict",
    5: "moon",
}


@dataclass
class ValidationIssue:
    level: str  # error | warning
    code: str
    message: str
    path: Optional[str] = None


@dataclass
class ValidationReport:
    ok: bool
    job_root: str
    errors: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "job_root": self.job_root,
            "errors": [asdict(e) for e in self.errors],
            "warnings": [asdict(w) for w in self.warnings],
            "stats": self.stats,
        }


def _issue(level: str, code: str, message: str, path: Optional[Path] = None) -> ValidationIssue:
    return ValidationIssue(level, code, message, str(path) if path else None)


def list_images(directory: Path) -> List[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def parse_label_line(line: str) -> Optional[Tuple[int, float, float, float, float]]:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        cls = int(float(parts[0]))
        vals = tuple(float(x) for x in parts[1:5])
        return cls, vals[0], vals[1], vals[2], vals[3]
    except ValueError:
        return None


def check_grayscale_style(img_bgr: np.ndarray) -> Tuple[bool, str]:
    """检查是否接近项目约定的 R=灰度、G=B=0 或纯灰度三通道。"""
    if img_bgr is None or img_bgr.ndim != 3:
        return False, "unreadable"
    r, g, b = img_bgr[:, :, 0], img_bgr[:, :, 1], img_bgr[:, :, 2]
    if g.max() == 0 and b.max() == 0:
        return True, "r_gray_gb_zero"
    if np.allclose(r, g) and np.allclose(g, b):
        return True, "gray_3ch"
    return False, "color_or_mixed"


def validate_job_input(
    job_root: Path,
    *,
    min_cali: int = 400,
    min_test: int = 100,
    nc: int = 6,
    expect_imgsz: int = 1280,
    class_names: Optional[Dict[int, str]] = None,
) -> ValidationReport:
    job_root = job_root.resolve()
    class_names = class_names or DEFAULT_CLASS_NAMES
    report = ValidationReport(ok=True, job_root=str(job_root))

    input_dir = job_root / "input"
    model_pt = input_dir / "model.pt"
    cali_dir = input_dir / "cali"
    test_img_dir = input_dir / "test" / "images"
    test_lab_dir = input_dir / "test" / "labels"

    # --- 目录结构 ---
    for path, code, msg in (
        (input_dir, "missing_input", "缺少 input/ 目录"),
        (cali_dir, "missing_cali", "缺少 input/cali/ 目录"),
        (test_img_dir, "missing_test_images", "缺少 input/test/images/ 目录"),
        (test_lab_dir, "missing_test_labels", "缺少 input/test/labels/ 目录"),
    ):
        if not path.is_dir():
            report.errors.append(_issue("error", code, msg, path))
            report.ok = False

    # --- 模型 ---
    if not model_pt.is_file():
        report.errors.append(_issue("error", "missing_model", "缺少 input/model.pt", model_pt))
        report.ok = False
    elif model_pt.stat().st_size < 1024:
        report.errors.append(_issue("error", "model_too_small", "model.pt 文件过小", model_pt))
        report.ok = False
    else:
        report.stats["model_pt_bytes"] = model_pt.stat().st_size

    cali_images = list_images(cali_dir)
    test_images = list_images(test_img_dir)
    report.stats["cali_count"] = len(cali_images)
    report.stats["test_image_count"] = len(test_images)

    if len(cali_images) < min_cali:
        report.errors.append(
            _issue("error", "cali_too_few", f"标定图 {len(cali_images)} 张，需要至少 {min_cali} 张", cali_dir)
        )
        report.ok = False

    if len(test_images) < min_test:
        report.errors.append(
            _issue(
                "error",
                "test_too_few",
                f"测试图 {len(test_images)} 张，需要至少 {min_test} 张",
                test_img_dir,
            )
        )
        report.ok = False

    # --- 标定与测试不得重复（现网 cali_data 与 quantize_test 文件名零交集）---
    cali_stems = {p.stem for p in cali_images}
    test_stems_for_overlap = {p.stem for p in test_images}
    overlap = cali_stems & test_stems_for_overlap
    report.stats["cali_test_overlap_count"] = len(overlap)
    if overlap:
        sample = ", ".join(sorted(overlap)[:5])
        report.errors.append(
            _issue(
                "error",
                "cali_test_overlap",
                f"标定与测试有 {len(overlap)} 张重复（文件名 stem 相同，示例: {sample}）",
                test_img_dir,
            )
        )
        report.ok = False

    if len(test_images) >= min_test and len(test_images) < 250:
        report.warnings.append(
            _issue(
                "warning",
                "test_below_recommended",
                f"测试图 {len(test_images)} 张；现网 quantize_test 约 281 张，建议 ≥250",
                test_img_dir,
            )
        )

    if len(cali_images) >= min_cali and len(cali_images) < 450:
        report.warnings.append(
            _issue(
                "warning",
                "cali_below_recommended",
                f"标定图 {len(cali_images)} 张；现网 cali_data 为 496 张全量参与量化，建议 450~500",
                cali_dir,
            )
        )

    # --- 同名配对 ---
    test_stems = test_stems_for_overlap
    label_files = sorted(test_lab_dir.glob("*.txt")) if test_lab_dir.is_dir() else []
    label_stems = {p.stem for p in label_files}
    paired = test_stems & label_stems
    img_only = test_stems - label_stems
    lab_only = label_stems - test_stems

    report.stats["test_label_paired"] = len(paired)
    report.stats["test_image_without_label"] = len(img_only)
    report.stats["test_label_without_image"] = len(lab_only)

    if img_only:
        sample = ", ".join(sorted(img_only)[:5])
        report.errors.append(
            _issue(
                "error",
                "label_missing",
                f"{len(img_only)} 张测试图无对应标签（示例: {sample}）",
                test_lab_dir,
            )
        )
        report.ok = False

    if lab_only:
        sample = ", ".join(sorted(lab_only)[:5])
        report.warnings.append(
            _issue(
                "warning",
                "orphan_label",
                f"{len(lab_only)} 个标签无对应测试图（示例: {sample}）",
                test_lab_dir,
            )
        )

    # --- 抽样检查图片与标签 ---
    class_counts: Dict[int, int] = {}
    size_set: Set[Tuple[int, int]] = set()
    gray_ok = 0
    gray_warn = 0
    sample_n = min(30, len(test_images))

    for img_path in test_images[:sample_n]:
        img = cv2.imread(str(img_path))
        if img is None:
            report.errors.append(_issue("error", "image_unreadable", "无法解码图片", img_path))
            report.ok = False
            continue
        h, w = img.shape[:2]
        size_set.add((w, h))
        if w != expect_imgsz or h != expect_imgsz:
            pass  # 统一在循环后给一条 warning
        is_gray, _ = check_grayscale_style(img)
        if is_gray:
            gray_ok += 1
        else:
            gray_warn += 1

    for lab_path in label_files[: min(500, len(label_files))]:
        for line in lab_path.read_text(encoding="utf-8").splitlines():
            parsed = parse_label_line(line)
            if parsed is None:
                if line.strip():
                    report.warnings.append(
                        _issue("warning", "label_bad_line", f"标签行格式异常: {line[:40]}", lab_path)
                    )
                continue
            cls, cx, cy, w, h = parsed
            class_counts[cls] = class_counts.get(cls, 0) + 1
            if cls < 0 or cls >= nc:
                report.errors.append(
                    _issue("error", "class_out_of_range", f"class_id={cls} 超出 nc={nc}", lab_path)
                )
                report.ok = False
            for v in (cx, cy, w, h):
                if v < 0 or v > 1:
                    report.errors.append(
                        _issue("error", "label_not_normalized", "标签坐标应在 0~1（YOLO 归一化）", lab_path)
                    )
                    report.ok = False
                    break

    report.stats["class_counts_in_labels"] = dict(sorted(class_counts.items()))
    report.stats["image_sizes_sample"] = [list(s) for s in sorted(size_set)]
    report.stats["grayscale_sample_ok"] = gray_ok
    report.stats["grayscale_sample_warn"] = gray_warn

    if size_set and any(s != (expect_imgsz, expect_imgsz) for s in size_set):
        report.warnings.append(
            _issue(
                "warning",
                "image_size_not_1280",
                f"部分测试图尺寸不是 {expect_imgsz}x{expect_imgsz}，流水线将自动 resize",
                test_img_dir,
            )
        )

    if gray_warn > 0:
        report.warnings.append(
            _issue(
                "warning",
                "not_grayscale",
                f"抽样 {sample_n} 张中有 {gray_warn} 张非标准黑白格式（建议 R=灰度且 G=B=0）；将强制转灰度",
                test_img_dir,
            )
        )

    # cali 抽样
    for cp in cali_images[:5]:
        cim = cv2.imread(str(cp))
        if cim is None:
            report.errors.append(_issue("error", "cali_unreadable", "标定图无法读取", cp))
            report.ok = False

    report.stats["nc"] = nc
    report.stats["class_names"] = class_names
    return report


def save_report(report: ValidationReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
