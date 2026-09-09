#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU readiness checks used before a quantization pipeline is started."""

from __future__ import annotations

import csv
import os
import subprocess
from dataclasses import asdict, dataclass
from typing import Any, Optional


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, value)


@dataclass(frozen=True)
class GPUReadiness:
    ready: bool
    reason: str
    required: bool
    min_free_mib: int
    max_utilization_percent: int
    index: Optional[int] = None
    uuid: Optional[str] = None
    name: Optional[str] = None
    memory_total_mib: Optional[int] = None
    memory_free_mib: Optional[int] = None
    utilization_percent: Optional[int] = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def poll_seconds() -> int:
    return _env_int("GPU_READY_POLL_SECONDS", 15, minimum=1)


def _result(*, ready: bool, reason: str, error: Optional[str] = None, **device: Any) -> GPUReadiness:
    return GPUReadiness(
        ready=ready,
        reason=reason,
        required=_env_bool("GPU_READINESS_REQUIRED", True),
        min_free_mib=_env_int("GPU_MIN_FREE_MIB", 24576),
        max_utilization_percent=_env_int("GPU_MAX_UTILIZATION_PERCENT", 90),
        error=error,
        **device,
    )


def probe_gpu() -> GPUReadiness:
    """Check the first visible GPU, which is logical CUDA device 0 in the app."""
    required = _env_bool("GPU_READINESS_REQUIRED", True)
    if not required:
        return _result(ready=True, reason="GPU readiness gate is disabled")

    fields = "index,uuid,name,memory.total,memory.free,utilization.gpu"
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={fields}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return _result(ready=False, reason="GPU query failed", error=str(exc))

    if completed.returncode != 0:
        error = (completed.stderr or completed.stdout or "nvidia-smi failed").strip()[-1000:]
        return _result(ready=False, reason="GPU query failed", error=error)

    rows = [row for row in csv.reader(completed.stdout.splitlines()) if row]
    if not rows:
        return _result(ready=False, reason="No GPU is visible to the API container")

    row = [item.strip() for item in rows[0]]
    if len(row) != 6:
        return _result(
            ready=False,
            reason="Unexpected nvidia-smi output",
            error=",".join(row)[-1000:],
        )

    try:
        index = int(row[0])
        total_mib = int(row[3])
        free_mib = int(row[4])
        utilization = int(row[5])
    except ValueError as exc:
        return _result(ready=False, reason="Invalid GPU metrics", error=str(exc))

    device = {
        "index": index,
        "uuid": row[1],
        "name": row[2],
        "memory_total_mib": total_mib,
        "memory_free_mib": free_mib,
        "utilization_percent": utilization,
    }
    min_free_mib = _env_int("GPU_MIN_FREE_MIB", 24576)
    max_utilization = _env_int("GPU_MAX_UTILIZATION_PERCENT", 90)
    if free_mib < min_free_mib:
        return _result(
            ready=False,
            reason=f"GPU free memory {free_mib} MiB is below the {min_free_mib} MiB threshold",
            **device,
        )
    if utilization > max_utilization:
        return _result(
            ready=False,
            reason=f"GPU utilization {utilization}% exceeds the {max_utilization}% threshold",
            **device,
        )
    return _result(ready=True, reason="GPU is ready", **device)
