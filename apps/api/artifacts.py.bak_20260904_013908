#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""任务指标、清单、评测样例图（JSON / 文件，带路径穿越保护）。"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import HTTPException
from fastapi.responses import FileResponse, PlainTextResponse

from apps.api._paths import setup_pipeline_imports

setup_pipeline_imports()

from bundle_inventory import format_bytes, inventory_job_dir, inventory_zip  # noqa: E402
from constants import IMAGE_EXTS  # noqa: E402
from job_config import JobConfig  # noqa: E402
from shared_datasets import load_registry  # noqa: E402

EVAL_NAMES = ("pt_eval", "onnx_eval", "fpga_eval")
CASE_SUBDIRS = ("error_cases", "success_cases")
_SAFE_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,255}$")
_IMAGE_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}


def _load_json(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _attach_overlay_bust(cfg: JobConfig, eval_name: str, subdir: str, cases: list) -> list:
    out = []
    for case in cases:
        c = dict(case)
        name = c.get("overlay") or ""
        if name:
            op = cfg.results_dir / eval_name / subdir / name
            if op.is_file():
                c["overlay_v"] = int(op.stat().st_mtime)
        out.append(c)
    return out


def _load_overlay_cases(cfg: JobConfig, name: str, kind: str) -> list:
    data = _load_json(cfg.results_dir / name / f"{kind}_cases.json")
    if not data:
        return []
    return _attach_overlay_bust(cfg, name, f"{kind}_cases", data.get("cases", []))


def _load_error_groups(cfg: JobConfig, name: str) -> dict:
    empty: Dict[str, list] = {"class_error": [], "fp": [], "fn": []}
    data = _load_json(cfg.results_dir / name / "error_cases.json")
    if not data:
        return empty
    groups = data.get("groups")
    if groups:
        return {
            "class_error": _attach_overlay_bust(cfg, name, "error_cases", groups.get("class_error", [])),
            "fp": _attach_overlay_bust(cfg, name, "error_cases", groups.get("fp", [])),
            "fn": _attach_overlay_bust(cfg, name, "error_cases", groups.get("fn", [])),
        }
    for case in data.get("cases", []):
        if case.get("class_error", 0) > 0:
            empty["class_error"].append(case)
        if case.get("FP", 0) > 0:
            empty["fp"].append(case)
        if case.get("FN", 0) > 0:
            empty["fn"].append(case)
    return {
        "class_error": _attach_overlay_bust(cfg, name, "error_cases", empty["class_error"]),
        "fp": _attach_overlay_bust(cfg, name, "error_cases", empty["fp"]),
        "fn": _attach_overlay_bust(cfg, name, "error_cases", empty["fn"]),
    }


def eval_dataset_desc(cfg: JobConfig) -> dict:
    images_dir = cfg.test_images_dir
    labels_dir = cfg.test_labels_dir
    n_images = 0
    if images_dir.is_dir():
        n_images = len([p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])

    inst: Counter = Counter()
    imgs_with: Counter = Counter()
    if labels_dir.is_dir():
        for lab in labels_dir.iterdir():
            if lab.suffix.lower() != ".txt":
                continue
            seen = set()
            for line in lab.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    cid = int(float(parts[0]))
                except ValueError:
                    continue
                inst[cid] += 1
                seen.add(cid)
            for cid in seen:
                imgs_with[cid] += 1

    reg_meta: dict = {}
    if cfg.test_dataset_id:
        reg_meta = load_registry().get("test", {}).get(cfg.test_dataset_id, {}) or {}
    name_map = {int(k): str(v) for k, v in (reg_meta.get("class_names") or {}).items()}
    class_rows = []
    for cid in sorted(set(inst) | set(range(cfg.nc))):
        class_rows.append(
            {
                "id": cid,
                "name": name_map.get(cid, f"class_{cid}"),
                "instances": int(inst.get(cid, 0)),
                "images": int(imgs_with.get(cid, 0)),
            }
        )
    n_eval = None
    for summary_name in EVAL_NAMES:
        sp = cfg.results_dir / summary_name / "summary.json"
        data = _load_json(sp)
        if data and data.get("n_images") is not None:
            n_eval = data.get("n_images")
            break

    return {
        "dataset_id": cfg.test_dataset_id or "(任务内测试集)",
        "dataset_name": reg_meta.get("display_name") or cfg.test_dataset_id or "未命名",
        "n_images": n_images,
        "n_eval": int(n_eval) if n_eval is not None else n_images,
        "nc": cfg.nc,
        "classes": class_rows,
        "cali_dataset_id": cfg.cali_dataset_id or "",
    }


def metrics_payload(cfg: JobConfig) -> dict[str, Any]:
    def load_summary(name: str) -> Optional[dict]:
        return _load_json(cfg.results_dir / name / "summary.json")

    return {
        "task_id": cfg.job_id,
        "display_name": cfg.display_name,
        "preprocess_mode": cfg.preprocess_mode,
        "pt": load_summary("pt_eval"),
        "onnx": load_summary("onnx_eval"),
        "fpga": load_summary("fpga_eval"),
        "pt_success": _load_overlay_cases(cfg, "pt_eval", "success"),
        "pt_errors": _load_error_groups(cfg, "pt_eval"),
        "onnx_errors": _load_error_groups(cfg, "onnx_eval"),
        "fpga_errors": _load_error_groups(cfg, "fpga_eval"),
        "eval_data": eval_dataset_desc(cfg),
    }


def bundle_inventory_payload(cfg: JobConfig) -> dict[str, Any]:
    zp = cfg.bundle_zip_path()
    if zp.is_file():
        inv = inventory_zip(zp)
        source_label = f"zip: {zp.name}"
        has_zip = True
    else:
        inv = inventory_job_dir(
            cfg.job_root,
            onnx_name=cfg.onnx_name,
            deliverable_only=True,
            workspace_dir=cfg.workspace_dir,
            fpga_test_pack_dir=cfg.fpga_test_pack_dir,
        )
        source_label = "成果目录（zip 尚未生成）"
        has_zip = False
    for row in inv.get("by_top_level", []):
        row["size_h"] = format_bytes(row["bytes"])
    for row in inv.get("largest_files", []):
        row["size_h"] = format_bytes(row["bytes"])
    inv["total_h"] = format_bytes(inv.get("total_bytes", 0))
    inv["task_id"] = cfg.job_id
    inv["source_label"] = source_label
    inv["has_zip"] = has_zip
    return inv


def input_manifest_response(cfg: JobConfig):
    path = cfg.input_manifest_path
    if not path.is_file():
        raise HTTPException(status_code=404, detail="input_manifest 不存在")
    return PlainTextResponse(path.read_text(encoding="utf-8"), media_type="text/markdown; charset=utf-8")


def overlay_file_response(cfg: JobConfig, eval_name: str, subdir: str, filename: str) -> FileResponse:
    if eval_name not in EVAL_NAMES or subdir not in CASE_SUBDIRS:
        raise HTTPException(status_code=404, detail="unknown eval path")
    if not _SAFE_FILENAME.fullmatch(filename) or ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="invalid filename")
    root = (cfg.results_dir / eval_name / subdir).resolve()
    path = (root / filename).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid filename")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="image not found")
    media = _IMAGE_MIME.get(path.suffix.lower(), "application/octet-stream")
    return FileResponse(
        path,
        media_type=media,
        filename=path.name,
        headers={"Cache-Control": "no-cache, max-age=0, must-revalidate"},
    )
