#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 input_manifest.md（WEB_DESIGN.md §12.3）。"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from constants import IMAGE_EXTS
from job_config import JobConfig


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _list_names(directory: Path, sidecar: Path) -> str:
    if not directory.is_dir():
        return "（目录不存在）"
    names = sorted(p.name for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTS or p.suffix == ".txt")
    sidecar.write_text("\n".join(names) + "\n", encoding="utf-8")
    if len(names) <= 20:
        return "\n".join(f"- {n}" for n in names)
    preview = "\n".join(f"- {n}" for n in names[:10])
    return f"{preview}\n- ... 共 {len(names)} 个，完整列表见 `{sidecar.name}`"


def write_input_manifest(cfg: JobConfig, *, validation_ok: bool | None = None) -> Path:
    cfg.job_root.mkdir(parents=True, exist_ok=True)
    pt_path = cfg.model_pt
    pt_name = pt_path.name if pt_path.is_file() else "model.pt"
    pt_md5 = file_md5(pt_path) if pt_path.is_file() else "N/A"

    cali_sidecar = cfg.job_root / "input_cali_files.txt"
    test_sidecar = cfg.job_root / "input_test_files.txt"
    cali_count = len([p for p in cfg.cali_dir.glob("*") if p.suffix.lower() in IMAGE_EXTS]) if cfg.cali_dir.is_dir() else 0
    test_count = len([p for p in cfg.test_images_dir.glob("*") if p.suffix.lower() in IMAGE_EXTS]) if cfg.test_images_dir.is_dir() else 0

    val_line = "unknown"
    if validation_ok is not None:
        val_line = str(validation_ok).lower()
    elif cfg.input_validation_path.is_file():
        import json
        val_line = str(json.loads(cfg.input_validation_path.read_text()).get("ok", "unknown")).lower()

    md = f"""# 任务输入清单 — {cfg.display_name or cfg.job_id}

- **任务 ID**: {cfg.job_id}
- **显示名**: {cfg.display_name}
- **创建时间**: {datetime.now(timezone.utc).isoformat()}

## 模型

- **文件名**: {pt_name}
- **MD5**: `{pt_md5}`

## 标定集

- **来源**: {"共享数据集 `" + cfg.cali_dataset_id + "`" if cfg.cali_dataset_id else "本任务 input/cali/"}
- **张数**: {cali_count}
{_list_names(cfg.cali_dir, cali_sidecar)}

## 测试集

- **来源**: {"共享数据集 `" + cfg.test_dataset_id + "`（含 fpga_test_pack 子目录）" if cfg.test_dataset_id else "本任务 input/test/"}
- **张数**: {test_count}
{_list_names(cfg.test_images_dir, test_sidecar)}

## 检校

- **input_validation.json**: ok={val_line}
"""
    cfg.input_manifest_path.write_text(md, encoding="utf-8")
    return cfg.input_manifest_path
