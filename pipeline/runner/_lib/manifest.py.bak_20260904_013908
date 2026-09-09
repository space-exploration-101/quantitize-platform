#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""任务 manifest：步骤状态持久化。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


PIPELINE_STEPS: List[str] = [
    "pt_eval",
    "quantize",
    "onnx_eval",
    "fpga_test_pack",
    "fpga_eval",
    "generate_bin",
    "merge_bin",
    "bundle",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_manifest(path: Path) -> Dict[str, Any]:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "steps": {name: {"status": "pending"} for name in PIPELINE_STEPS},
        "updated_at": _now_iso(),
    }


def save_manifest(path: Path, data: Dict[str, Any]) -> None:
    data["updated_at"] = _now_iso()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def set_step_status(
    manifest: Dict[str, Any],
    step: str,
    status: str,
    *,
    message: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    if step not in manifest.setdefault("steps", {}):
        manifest["steps"][step] = {}
    entry = manifest["steps"][step]
    entry["status"] = status
    now = _now_iso()
    entry["updated_at"] = now
    if status == "running":
        entry["started_at"] = now
        entry.pop("duration_seconds", None)
    elif status in {"completed", "failed"}:
        started = entry.get("started_at")
        if started:
            try:
                entry["duration_seconds"] = round(
                    (datetime.fromisoformat(now) - datetime.fromisoformat(started)).total_seconds(),
                    3,
                )
            except (TypeError, ValueError):
                pass
    if status == "completed" and message is None:
        entry.pop("message", None)
    elif message is not None:
        entry["message"] = message
    if extra:
        entry.update(extra)
