#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""后台单任务 Worker：启动 Runner，同一时刻只跑一个任务。"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Optional

from apps.api._paths import setup_pipeline_imports
from apps.api.gpu_readiness import GPUReadiness, poll_seconds, probe_gpu

setup_pipeline_imports()

from env import run_script  # noqa: E402
from manifest import load_manifest, save_manifest  # noqa: E402
from script_registry import PlatformScripts, platform_script  # noqa: E402

_lock = threading.Lock()
_state_lock = threading.Lock()
_cancel_wait = threading.Event()
_current_job: Optional[str] = None
_last_error: Optional[str] = None
_worker_status = "idle"
_last_gpu: Optional[GPUReadiness] = None


def is_busy() -> bool:
    return _lock.locked()


def current_job_id() -> Optional[str]:
    with _state_lock:
        return _current_job


def last_error() -> Optional[str]:
    with _state_lock:
        return _last_error


def worker_status() -> str:
    with _state_lock:
        return _worker_status


def last_gpu_readiness() -> dict[str, Any]:
    with _state_lock:
        current = _last_gpu
    if current is None:
        current = probe_gpu()
    return current.to_dict()


def _set_runtime_state(
    *,
    status: str,
    job_id: Optional[str],
    error: Optional[str] = None,
    gpu: Optional[GPUReadiness] = None,
) -> None:
    global _current_job, _last_error, _worker_status, _last_gpu
    with _state_lock:
        _current_job = job_id
        _last_error = error
        _worker_status = status
        if gpu is not None:
            _last_gpu = gpu


def _set_job_status(
    job_dir: Path,
    status: str,
    *,
    gpu: Optional[GPUReadiness] = None,
    error: Optional[str] = None,
) -> None:
    manifest_path = job_dir / "manifest.json"
    manifest = load_manifest(manifest_path)
    manifest["status"] = status
    if gpu is not None:
        manifest["gpu_readiness"] = gpu.to_dict()
    if error:
        manifest["error"] = error
    elif status in {"waiting_gpu", "running", "pending"}:
        manifest.pop("error", None)
    save_manifest(manifest_path, manifest)


def start_pipeline(job_dir: Path) -> Optional[dict[str, Any]]:
    """Start one pipeline, waiting for the selected GPU when necessary."""
    job_dir = job_dir.resolve()
    if not _lock.acquire(blocking=False):
        return None

    _cancel_wait.clear()
    initial_gpu = probe_gpu()
    initial_status = "running" if initial_gpu.ready else "waiting_gpu"
    _set_job_status(job_dir, initial_status, gpu=initial_gpu)
    _set_runtime_state(status=initial_status, job_id=job_dir.name, gpu=initial_gpu)

    def _run() -> None:
        try:
            gpu = initial_gpu
            while not gpu.ready:
                if _cancel_wait.wait(poll_seconds()):
                    _set_job_status(job_dir, "pending")
                    _set_runtime_state(status="idle", job_id=None, gpu=gpu)
                    return
                gpu = probe_gpu()
                _set_job_status(job_dir, "waiting_gpu", gpu=gpu)
                _set_runtime_state(status="waiting_gpu", job_id=job_dir.name, gpu=gpu)

            _set_job_status(job_dir, "running", gpu=gpu)
            _set_runtime_state(status="running", job_id=job_dir.name, gpu=gpu)
            runner = platform_script(PlatformScripts.RUNNER)
            r = run_script(runner, ["--job-dir", str(job_dir)])
            if r.returncode != 0:
                error = (r.stderr or r.stdout or "")[-2000:]
                _set_runtime_state(status="failed", job_id=job_dir.name, error=error, gpu=gpu)
        except Exception as e:
            error = str(e)
            _set_job_status(job_dir, "failed", error=error)
            _set_runtime_state(status="failed", job_id=job_dir.name, error=error)
        finally:
            if worker_status() != "idle":
                _set_runtime_state(
                    status="idle",
                    job_id=None,
                    error=last_error(),
                )
            _lock.release()

    threading.Thread(target=_run, daemon=True).start()
    return {"status": initial_status, "gpu_readiness": initial_gpu.to_dict()}


def cancel_waiting(job_id: str) -> bool:
    """Cancel a waiting job without interrupting an already-running pipeline."""
    if current_job_id() != job_id or worker_status() != "waiting_gpu":
        return False
    _cancel_wait.set()
    return True
