#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantitize Platform JSON API。

查询、启动、下载见 P2；创建任务（ZIP / shared）见 P3。
不渲染 HTML，不导入旧 Web / 旧 quantitize 运行时。
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from apps.api._paths import setup_pipeline_imports
from apps.api.artifacts import (
    bundle_inventory_payload,
    input_manifest_response,
    metrics_payload,
    overlay_file_response,
)
from apps.api.create_task import (
    DEFAULT_PREPROCESS,
    MAX_ZIP_BYTES,
    create_from_shared,
    create_from_zip,
    read_upload,
)
from apps.api.gpu_readiness import probe_gpu
from apps.api.worker import (
    cancel_waiting,
    current_job_id,
    is_busy,
    last_error,
    last_gpu_readiness,
    start_pipeline,
    worker_status as worker_runtime_status,
)

setup_pipeline_imports()

from job_config import OUTPUT_DATA_ROOT, JobConfig  # noqa: E402
from manifest import PIPELINE_STEPS, load_manifest, save_manifest  # noqa: E402
from shared_datasets import (  # noqa: E402
    list_cali_datasets,
    list_dataset_catalog,
    list_test_datasets,
)

# 单层目录名：禁止穿越。允许 Unicode（历史任务可能把中文写进目录名）。
_TASK_ID_RE = re.compile(r"^(?!^\.\.?$)[^/\\\x00]{1,128}$")

app = FastAPI(title="Quantitize Platform API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _task_status(manifest: dict) -> str:
    if manifest.get("status") in ("waiting_gpu", "running", "failed", "completed"):
        return manifest["status"]
    steps = manifest.get("steps", {})
    if any(s.get("status") == "running" for s in steps.values()):
        return "running"
    if any(s.get("status") == "failed" for s in steps.values()):
        return "failed"
    if steps and all(s.get("status") == "completed" for s in steps.values()):
        return "completed"
    return "pending"


def _dataset_payload(entry) -> dict[str, Any]:
    return {
        "id": entry.id,
        "name": entry.display_name,
        "display_name": entry.display_name,
        "kind": entry.kind,
        "path": entry.rel_path,
        "image_count": entry.image_count,
        "note": entry.note,
    }


def _job_dir(task_id: str) -> Path:
    if not _TASK_ID_RE.fullmatch(task_id):
        raise HTTPException(status_code=400, detail="invalid task id")
    root = Path(OUTPUT_DATA_ROOT).resolve()
    job = (root / task_id).resolve()
    if job.parent != root:
        raise HTTPException(status_code=400, detail="invalid task id")
    if not job.is_dir():
        raise HTTPException(status_code=404, detail="task not found")
    return job


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "ok": True,
        "service": "quantitize-platform-api",
        "health": "/health",
        "docs": "/docs",
        "tasks": "/api/tasks",
        "datasets": "/api/datasets",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "busy": is_busy(),
        "worker_status": worker_runtime_status(),
        "current_job": current_job_id(),
    }


@app.get("/ready")
def ready():
    gpu = probe_gpu().to_dict()
    return JSONResponse({"ok": gpu["ready"], "gpu": gpu}, status_code=200 if gpu["ready"] else 503)


@app.get("/api/gpu")
def gpu_state() -> dict[str, Any]:
    gpu = probe_gpu().to_dict()
    return {"ready": gpu["ready"], "gpu": gpu}


@app.get("/api/worker")
def worker_state() -> dict[str, Any]:
    return {
        "busy": is_busy(),
        "status": worker_runtime_status(),
        "current_job": current_job_id(),
        "last_error": last_error(),
        "gpu_readiness": last_gpu_readiness(),
    }


@app.get("/api/datasets")
def datasets() -> dict[str, Any]:
    return {
        "cali": [_dataset_payload(d) for d in list_cali_datasets()],
        "test": [_dataset_payload(d) for d in list_test_datasets()],
        "catalog": [
            {
                "kind_label": row.kind_label,
                "download_kind": row.download_kind,
                "entry": _dataset_payload(row.entry),
            }
            for row in list_dataset_catalog()
        ],
    }


@app.get("/api/tasks")
def list_tasks() -> dict[str, Any]:
    rows = []
    root = Path(OUTPUT_DATA_ROOT)
    if root.is_dir():
        for p in sorted(root.iterdir(), reverse=True):
            if not p.is_dir():
                continue
            try:
                cfg = JobConfig.from_job_dir(p)
                manifest = load_manifest(cfg.manifest_path)
                rows.append(
                    {
                        "task_id": cfg.job_id,
                        "display_name": cfg.display_name,
                        "status": _task_status(manifest),
                        "has_zip": cfg.bundle_zip_path().is_file(),
                        "updated": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                    }
                )
            except Exception:
                continue
    return {"tasks": rows}


@app.get("/api/tasks/{task_id}")
def get_task(task_id: str) -> dict[str, Any]:
    cfg = JobConfig.from_job_dir(_job_dir(task_id))
    manifest = load_manifest(cfg.manifest_path)
    return {
        "task_id": cfg.job_id,
        "display_name": cfg.display_name,
        "onnx_name": cfg.onnx_name,
        "preprocess_mode": cfg.preprocess_mode,
        "status": _task_status(manifest),
        "manifest": manifest,
        "steps": list(PIPELINE_STEPS),
        "busy": is_busy(),
        "current_job": current_job_id(),
        "worker_error": last_error(),
        "task_error": manifest.get("error"),
        "worker_status": worker_runtime_status(),
        "gpu_readiness": manifest.get("gpu_readiness") or last_gpu_readiness(),
        "has_zip": cfg.bundle_zip_path().is_file(),
        "cali_dataset_id": cfg.cali_dataset_id,
        "test_dataset_id": cfg.test_dataset_id,
    }


@app.post("/api/tasks")
async def create_task_zip(
    display_name: str = Form(""),
    onnx_name: str = Form(""),
    preprocess_mode: str = Form(DEFAULT_PREPROCESS),
    archive: Optional[UploadFile] = File(None),
    zip_file: Optional[UploadFile] = File(None),
):
    if is_busy():
        raise HTTPException(status_code=409, detail="已有任务在运行")
    upload = archive or zip_file
    content = await read_upload(upload, limit=MAX_ZIP_BYTES, what="archive")
    status, payload = create_from_zip(
        display_name=display_name,
        onnx_name=onnx_name,
        preprocess_mode=preprocess_mode,
        content=content,
    )
    return JSONResponse(payload, status_code=status)


@app.post("/api/tasks/shared")
async def create_task_shared(
    display_name: str = Form(""),
    onnx_name: str = Form(""),
    preprocess_mode: str = Form(DEFAULT_PREPROCESS),
    cali_dataset_id: str = Form(...),
    test_dataset_id: str = Form(...),
    imgsz: int = Form(1280),
    min_cali_images: Optional[int] = Form(None),
    min_test_images: Optional[int] = Form(None),
    nc: Optional[int] = Form(None),
    model: Optional[UploadFile] = File(None),
    model_pt: Optional[UploadFile] = File(None),
):
    if is_busy():
        raise HTTPException(status_code=409, detail="已有任务在运行")
    upload = model or model_pt
    content = await read_upload(upload, limit=MAX_ZIP_BYTES, what="model")
    status, payload = create_from_shared(
        display_name=display_name,
        onnx_name=onnx_name,
        preprocess_mode=preprocess_mode,
        cali_dataset_id=cali_dataset_id,
        test_dataset_id=test_dataset_id,
        imgsz=imgsz,
        model_bytes=content,
        min_cali_images=min_cali_images,
        min_test_images=min_test_images,
        nc=nc,
    )
    return JSONResponse(payload, status_code=status)


@app.get("/api/tasks/{task_id}/metrics")
def task_metrics(task_id: str) -> dict[str, Any]:
    cfg = JobConfig.from_job_dir(_job_dir(task_id))
    return metrics_payload(cfg)


@app.get("/api/tasks/{task_id}/bundle_inventory")
def task_bundle_inventory(task_id: str) -> dict[str, Any]:
    cfg = JobConfig.from_job_dir(_job_dir(task_id))
    return bundle_inventory_payload(cfg)


@app.get("/api/tasks/{task_id}/input_manifest")
def task_input_manifest(task_id: str):
    cfg = JobConfig.from_job_dir(_job_dir(task_id))
    return input_manifest_response(cfg)


@app.get("/api/tasks/{task_id}/results/{eval_name}/error_cases/{filename}")
def task_error_case_image(task_id: str, eval_name: str, filename: str):
    cfg = JobConfig.from_job_dir(_job_dir(task_id))
    return overlay_file_response(cfg, eval_name, "error_cases", filename)


@app.get("/api/tasks/{task_id}/results/{eval_name}/success_cases/{filename}")
def task_success_case_image(task_id: str, eval_name: str, filename: str):
    cfg = JobConfig.from_job_dir(_job_dir(task_id))
    return overlay_file_response(cfg, eval_name, "success_cases", filename)


@app.post("/api/tasks/{task_id}/start")
def start_task(task_id: str) -> dict[str, Any]:
    if is_busy():
        raise HTTPException(status_code=409, detail="已有任务在运行")
    cfg = JobConfig.from_job_dir(_job_dir(task_id))
    if not cfg.model_pt.is_file():
        raise HTTPException(status_code=400, detail="缺少 model.pt")
    result = start_pipeline(cfg.job_root)
    if result is None:
        raise HTTPException(status_code=409, detail="无法启动")
    return {"ok": True, "task_id": task_id, **result}


@app.post("/api/tasks/{task_id}/cancel-wait")
def cancel_task_wait(task_id: str) -> dict[str, Any]:
    _job_dir(task_id)
    if not cancel_waiting(task_id):
        raise HTTPException(status_code=409, detail="任务当前不在等待 GPU")
    return {"ok": True, "task_id": task_id, "status": "pending"}


@app.on_event("startup")
def resume_waiting_task() -> None:
    """Recover persisted worker state after an API container restart."""
    root = Path(OUTPUT_DATA_ROOT)
    if not root.is_dir() or is_busy():
        return
    waiting_job: Optional[Path] = None
    for job_dir in sorted((p for p in root.iterdir() if p.is_dir()), reverse=True):
        try:
            manifest = load_manifest(job_dir / "manifest.json")
            status = manifest.get("status")
            if status == "running":
                message = "API 容器在任务运行期间重启；请检查日志后重试流水线"
                manifest["status"] = "failed"
                manifest["error"] = message
                for step in (manifest.get("steps") or {}).values():
                    if step.get("status") == "running":
                        step["status"] = "failed"
                        step["message"] = message
                save_manifest(job_dir / "manifest.json", manifest)
            elif status == "waiting_gpu" and waiting_job is None:
                waiting_job = job_dir
        except Exception:
            continue
    if waiting_job is not None:
        start_pipeline(waiting_job)


@app.get("/api/tasks/{task_id}/download")
def download_bundle(task_id: str):
    cfg = JobConfig.from_job_dir(_job_dir(task_id))
    zp = cfg.bundle_zip_path()
    if not zp.is_file():
        raise HTTPException(status_code=404, detail="bundle 不存在")
    return FileResponse(zp, filename=zp.name, media_type="application/zip")
