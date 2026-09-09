#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""产物 zip / 目录体积分析。"""

from __future__ import annotations

import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _dir_inventory(root: Path, prefix: str = "") -> Tuple[List[Dict[str, Any]], int]:
    items: List[Dict[str, Any]] = []
    total = 0
    if not root.is_dir():
        return items, 0
    for path in root.rglob("*"):
        if path.is_file():
            size = path.stat().st_size
            rel = str(path.relative_to(root))
            arc = f"{prefix}/{rel}" if prefix else rel
            items.append({"path": arc, "bytes": size})
            total += size
    return items, total


def inventory_zip(zip_path: Path) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    by_top: Dict[str, int] = defaultdict(int)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            items.append({"path": info.filename, "bytes": info.file_size})
            top = info.filename.split("/")[0] if "/" in info.filename else info.filename
            by_top[top] += info.file_size
    items.sort(key=lambda x: -x["bytes"])
    total = sum(x["bytes"] for x in items)
    return {
        "source": str(zip_path),
        "total_bytes": total,
        "file_count": len(items),
        "by_top_level": [
            {"name": k, "bytes": v, "pct": round(100 * v / total, 1) if total else 0}
            for k, v in sorted(by_top.items(), key=lambda x: -x[1])
        ],
        "largest_files": items[:40],
    }


def workspace_file_in_bundle(rel: str, onnx_name: str, *, slim: bool) -> bool:
    """workspace/ 下是否纳入成果 zip。rel 为相对 workspace 的路径。"""
    name = Path(rel).name
    if name == f"{onnx_name}.onnx":
        return False
    if name.endswith("_fp16.onnx") or name.endswith("_fp16_output.onnx"):
        return False
    if slim:
        # 精简包：排除 generate_bin 原始层 bin，保留 01/03 rename 后的每层 wt/bn 与 all_bin
        if rel.startswith("bin/"):
            return False
        if rel.startswith("info_txt/"):
            return False
    return True


def arcname_in_bundle(arcname: str, onnx_name: str, *, slim: bool) -> bool:
    if arcname.startswith("workspace/"):
        rel = arcname[len("workspace/") :]
        return workspace_file_in_bundle(rel, onnx_name, slim=slim)
    if arcname.startswith("fpga_test_pack/"):
        return False
    if slim:
        if arcname.startswith("workspace/bin/"):
            return False
        if arcname.startswith("workspace/info_txt/"):
            return False
        for s in ("_fp16.onnx", "_fp16_output.onnx"):
            if arcname.endswith(s):
                return False
    return True


def inventory_job_dir(
    job_root: Path,
    *,
    onnx_name: str = "",
    deliverable_only: bool = False,
    workspace_dir: Optional[Path] = None,
    fpga_test_pack_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    sections = [
        ("workspace", workspace_dir or job_root / "workspace"),
        ("fpga_test_pack", fpga_test_pack_dir or job_root / "fpga_test_pack"),
        ("results", job_root / "results"),
        ("logs", job_root / "logs"),
        ("input", job_root / "input"),
    ]
    all_items: List[Dict[str, Any]] = []
    by_top: Dict[str, int] = defaultdict(int)
    for prefix, path in sections:
        items, _ = _dir_inventory(path, prefix)
        all_items.extend(items)
        by_top[prefix] += sum(i["bytes"] for i in items)
    all_items.sort(key=lambda x: -x["bytes"])
    if deliverable_only and onnx_name:
        all_items = [
            i
            for i in all_items
            if arcname_in_bundle(i["path"], onnx_name, slim=True)
        ]
        by_top = defaultdict(int)
        for i in all_items:
            top = i["path"].split("/")[0]
            by_top[top] += i["bytes"]
    total = sum(by_top.values())
    return {
        "source": str(job_root),
        "total_bytes": total,
        "file_count": len(all_items),
        "by_top_level": [
            {"name": k, "bytes": v, "pct": round(100 * v / total, 1) if total else 0}
            for k, v in sorted(by_top.items(), key=lambda x: -x[1])
        ],
        "largest_files": all_items[:40],
    }


def format_bytes(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f} GB"
    if n >= 1_000_000:
        return f"{n / 1e6:.1f} MB"
    if n >= 1000:
        return f"{n / 1e3:.1f} KB"
    return f"{n} B"
