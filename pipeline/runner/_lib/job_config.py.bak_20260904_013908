#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Job 目录布局与路径（对应 WEB_DESIGN.md §6.1）。"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

_PIPELINE = Path(__file__).resolve().parents[2]
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))
from script_registry import (  # noqa: E402
    OUTPUT_DATA_ROOT,
    PLATFORM_ROOT,
    QUANTITIZE_DIR,
    TASK_SCRATCH_ROOT,
)


def ascii_task_slug(display_name: str, max_len: int = 40) -> str:
    """目录名只保留 ASCII，避免中文 display_name 生成无法被 API 校验的 task_id。"""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", display_name or "")[:max_len].strip("._-")
    return slug or "task"


@dataclass
class JobConfig:
    job_root: Path
    job_id: str
    onnx_name: str
    display_name: str = ""
    nc: int = 6
    imgsz: int = 1280
    conf: float = 0.25
    iou: float = 0.7
    max_det: int = 300
    min_test_images: int = 100
    min_cali_images: int = 400
    cali_dataset_id: str = ""
    test_dataset_id: str = ""
    preprocess_mode: str = "grayscale_uniform"
    use_scratch: bool = False

    @classmethod
    def from_job_dir(cls, job_dir: Path, **overrides) -> "JobConfig":
        job_dir = job_dir.resolve()
        meta_path = job_dir / "job_config.json"
        data = {}
        if meta_path.is_file():
            raw = meta_path.read_text(encoding="utf-8").strip()
            if raw:
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    data = {}
        data.update(overrides)
        job_id = data.get("job_id", job_dir.name)
        onnx_name = data.get("onnx_name", job_id)
        display_name = data.get("display_name", job_id)
        return cls(
            job_root=job_dir,
            job_id=job_id,
            onnx_name=onnx_name,
            display_name=display_name,
            nc=int(data.get("nc", 6)),
            imgsz=int(data.get("imgsz", 1280)),
            conf=float(data.get("conf", 0.25)),
            iou=float(data.get("iou", 0.7)),
            max_det=int(data.get("max_det", 300)),
            min_test_images=int(data.get("min_test_images", 100)),
            min_cali_images=int(data.get("min_cali_images", 400)),
            cali_dataset_id=str(data.get("cali_dataset_id", "")),
            test_dataset_id=str(data.get("test_dataset_id", "")),
            preprocess_mode=str(data.get("preprocess_mode", "grayscale_uniform")),
            use_scratch=bool(data.get("use_scratch", False)),
        )

    @classmethod
    def create_task(
        cls,
        display_name: str,
        onnx_name: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs,
    ) -> "JobConfig":
        OUTPUT_DATA_ROOT.mkdir(parents=True, exist_ok=True)
        if task_id is None:
            slug = ascii_task_slug(display_name)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            task_id = f"{ts}_{slug}"
            n = 2
            while (OUTPUT_DATA_ROOT / task_id).exists():
                task_id = f"{ts}_{slug}_{n}"
                n += 1
                if n > 99:
                    raise RuntimeError("unable to allocate unique task id")
        job_root = OUTPUT_DATA_ROOT / task_id
        if "use_scratch" not in kwargs:
            kwargs["use_scratch"] = TASK_SCRATCH_ROOT is not None
        cfg = cls(
            job_root=job_root,
            job_id=task_id,
            onnx_name=onnx_name or task_id,
            display_name=display_name,
            **{k: v for k, v in kwargs.items() if k in cls.__dataclass_fields__},
        )
        cfg.ensure_layout()
        cfg.save()
        return cfg

    def save(self) -> None:
        self.job_root.mkdir(parents=True, exist_ok=True)
        path = self.job_root / "job_config.json"
        payload = asdict(self)
        payload["job_root"] = str(self.job_root)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def ensure_layout(self) -> None:
        for d in (
            self.input_dir,
            self.workspace_dir,
            self.results_dir,
            self.logs_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
        self.fpga_test_pack_dir.mkdir(parents=True, exist_ok=True)
        if not self.cali_dataset_id:
            self.cali_dir.mkdir(parents=True, exist_ok=True)
        if not self.test_dataset_id:
            self.test_images_dir.mkdir(parents=True, exist_ok=True)
            self.test_labels_dir.mkdir(parents=True, exist_ok=True)

    @property
    def input_dir(self) -> Path:
        return self.job_root / "input"

    @property
    def model_pt(self) -> Path:
        return self.input_dir / "model.pt"

    @property
    def cali_dir(self) -> Path:
        return self.input_dir / "cali"

    @property
    def test_images_dir(self) -> Path:
        return self.input_dir / "test" / "images"

    @property
    def test_labels_dir(self) -> Path:
        return self.input_dir / "test" / "labels"

    @property
    def workspace_dir(self) -> Path:
        if self.use_scratch:
            return self.scratch_dir / "workspace"
        return self.job_root / "workspace"

    @property
    def fpga_test_pack_dir(self) -> Path:
        """FPGA 测试包按任务生成（由当前绑定的测试集 images 转换，不放在 shared_data）。"""
        if self.use_scratch:
            return self.scratch_dir / "fpga_test_pack"
        return self.job_root / "fpga_test_pack"

    @property
    def scratch_dir(self) -> Path:
        if not self.use_scratch:
            return self.job_root
        if TASK_SCRATCH_ROOT is None:
            raise RuntimeError(
                f"任务 {self.job_id} 需要 scratch，但未设置 TASK_SCRATCH_ROOT"
            )
        root = TASK_SCRATCH_ROOT.resolve()
        path = (root / self.job_id).resolve()
        if path.parent != root:
            raise RuntimeError(f"非法 scratch 任务路径: {path}")
        return path

    @property
    def results_dir(self) -> Path:
        return self.job_root / "results"

    @property
    def logs_dir(self) -> Path:
        return self.job_root / "logs"

    @property
    def manifest_path(self) -> Path:
        return self.job_root / "manifest.json"

    @property
    def input_manifest_path(self) -> Path:
        return self.job_root / "input_manifest.md"

    @property
    def input_validation_path(self) -> Path:
        return self.job_root / "input_validation.json"

    def fp16_onnx(self) -> Path:
        return self.workspace_dir / f"{self.onnx_name}_fp16.onnx"

    def quantized_onnx(self) -> Path:
        return self.workspace_dir / f"{self.onnx_name}.onnx"

    def output_onnx(self) -> Path:
        return self.workspace_dir / f"{self.onnx_name}_output.onnx"

    def layer_bin_dir(self) -> Path:
        return self.workspace_dir / "bin"

    def all_bin_dir(self) -> Path:
        return self.workspace_dir / "all_bin"

    def pt_eval_dir(self) -> Path:
        return self.results_dir / "pt_eval"

    def onnx_eval_dir(self) -> Path:
        return self.results_dir / "onnx_eval"

    def fpga_eval_dir(self) -> Path:
        return self.results_dir / "fpga_eval"

    def test_label_wh_ref(self) -> Optional[int]:
        if not self.test_dataset_id:
            return None
        from shared_datasets import test_dataset_label_wh_ref  # noqa: WPS433

        return test_dataset_label_wh_ref(self.test_dataset_id)

    def bundle_zip_path(self) -> Path:
        return self.job_root / f"{self.job_id}_quantized_bundle.zip"
