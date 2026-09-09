#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-task NVMe scratch lifecycle with verified /data3 failure archives."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

from job_config import JobConfig
from script_registry import TASK_SCRATCH_ROOT

DEFAULT_MIN_FREE_BYTES = 150 * 1024**3


def _min_free_bytes() -> int:
    raw = os.environ.get("TASK_SCRATCH_MIN_FREE_BYTES", "").strip()
    return int(raw) if raw else DEFAULT_MIN_FREE_BYTES


def _validated_scratch_dir(cfg: JobConfig) -> Path:
    if not cfg.use_scratch:
        return cfg.job_root.resolve()
    if TASK_SCRATCH_ROOT is None:
        raise RuntimeError("任务启用了 scratch，但 TASK_SCRATCH_ROOT 未配置")
    root = TASK_SCRATCH_ROOT.resolve()
    target = cfg.scratch_dir.resolve()
    if target.parent != root or target.name != cfg.job_id:
        raise RuntimeError(f"拒绝操作非法 scratch 路径: {target}")
    return target


def ensure_scratch_ready(cfg: JobConfig) -> dict[str, Any]:
    """Allocate the task scratch and enforce the configured free-space floor."""
    if not cfg.use_scratch:
        return {"enabled": False}
    target = _validated_scratch_dir(cfg)
    root = target.parent
    if not root.is_dir():
        raise RuntimeError(f"scratch 根目录不存在或未挂载: {root}")
    usage = shutil.disk_usage(root)
    minimum = _min_free_bytes()
    if usage.free < minimum:
        raise RuntimeError(
            f"scratch 可用空间不足: free={usage.free}, required={minimum}, root={root}"
        )
    target.mkdir(parents=False, exist_ok=True)
    cfg.workspace_dir.mkdir(parents=True, exist_ok=True)
    cfg.fpga_test_pack_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "task_id": cfg.job_id,
        "status": "allocated",
        "root": str(target),
        "free_bytes_at_start": usage.free,
        "min_free_bytes": minimum,
        "updated_at_epoch": time.time(),
    }
    (target / "scratch_state.json").write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {"enabled": True, **state}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory(root: Path) -> dict[str, Any]:
    files = []
    total = 0
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        stat = path.stat()
        size = stat.st_size
        total += size
        files.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": _sha256(path),
            }
        )
    return {
        "source": str(root),
        "file_count": len(files),
        "total_bytes": total,
        "files": files,
    }


def _remove_scratch(cfg: JobConfig) -> None:
    target = _validated_scratch_dir(cfg)
    if not cfg.use_scratch or not target.exists():
        return
    shutil.rmtree(target)


def archive_failed_scratch(cfg: JobConfig) -> dict[str, Any]:
    """Copy a complete failed scratch to the persistent job and verify hashes."""
    if not cfg.use_scratch:
        return {"status": "not_applicable"}
    source = _validated_scratch_dir(cfg)
    if not source.is_dir():
        return {"status": "missing", "source": str(source)}

    final = cfg.job_root / "failure_artifacts"
    partial = cfg.job_root / "failure_artifacts.partial"
    if final.is_dir():
        inv = _inventory(final)
        inv["status"] = "verified"
        inv["archive"] = str(final)
        (cfg.job_root / "failure_inventory.json").write_text(
            json.dumps(inv, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        _remove_scratch(cfg)
        return inv

    if partial.exists():
        shutil.rmtree(partial)
    shutil.copytree(source, partial, symlinks=False)

    source_inv = _inventory(source)
    archive_inv = _inventory(partial)
    source_rows = [(r["path"], r["bytes"], r["sha256"]) for r in source_inv["files"]]
    archive_rows = [(r["path"], r["bytes"], r["sha256"]) for r in archive_inv["files"]]
    if source_rows != archive_rows:
        raise RuntimeError("失败任务 scratch 归档校验不一致，保留源 scratch 和 partial")

    partial.replace(final)
    archive_inv.update(
        {
            "status": "verified",
            "source": str(source),
            "archive": str(final),
            "verified_at_epoch": time.time(),
        }
    )
    (cfg.job_root / "failure_inventory.json").write_text(
        json.dumps(archive_inv, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _remove_scratch(cfg)
    return archive_inv


def finalize_success_scratch(cfg: JobConfig) -> dict[str, Any]:
    """Delete scratch only after the persistent bundle passes a ZIP integrity check."""
    if not cfg.use_scratch:
        return {"status": "not_applicable"}
    bundle = cfg.bundle_zip_path()
    if not bundle.is_file():
        raise RuntimeError(f"正式 bundle 不存在，拒绝清理 scratch: {bundle}")
    with zipfile.ZipFile(bundle, "r") as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"bundle ZIP 校验失败，拒绝清理 scratch: {bad}")
    source = _validated_scratch_dir(cfg)
    existed = source.exists()
    _remove_scratch(cfg)
    return {
        "status": "cleaned" if existed else "already_clean",
        "source": str(source),
        "bundle": str(bundle),
        "verified_at_epoch": time.time(),
    }
