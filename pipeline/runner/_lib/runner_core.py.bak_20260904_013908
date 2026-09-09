#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""8 步流水线编排核心（subprocess 调引擎脚本，路径见 script_registry）。"""

from __future__ import annotations

import sys
from pathlib import Path

_PLATFORM_LIB = Path(__file__).resolve().parent
if str(_PLATFORM_LIB) not in sys.path:
    sys.path.insert(0, str(_PLATFORM_LIB))

_PIPELINE_DIR = _PLATFORM_LIB.parents[1]  # pipeline/
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

_ENGINE_DIR = _PIPELINE_DIR / "engine"
if str(_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(_ENGINE_DIR))

from script_registry import EngineScripts, PlatformScripts, engine_script, platform_script  # noqa: E402

from env import run_script  # noqa: E402
from input_manifest import write_input_manifest  # noqa: E402
from job_config import JobConfig  # noqa: E402
from manifest import load_manifest, save_manifest, set_step_status  # noqa: E402
from scratch_lifecycle import (  # noqa: E402
    archive_failed_scratch,
    ensure_scratch_ready,
    finalize_success_scratch,
)


def _log(cfg: JobConfig, step: str) -> Path:
    return cfg.logs_dir / f"{step}.log"


def step_pt_eval(cfg: JobConfig, manifest: dict) -> None:
    set_step_status(manifest, "pt_eval", "running")
    script = platform_script(PlatformScripts.PT_EVAL)
    r = run_script(
        script,
        ["--job-dir", str(cfg.job_root)],
        log_path=_log(cfg, "pt_eval"),
    )
    if r.returncode != 0:
        set_step_status(manifest, "pt_eval", "failed", message=r.stderr[-2000:] if r.stderr else "")
        raise RuntimeError(f"pt_eval 失败: exit {r.returncode}")
    set_step_status(manifest, "pt_eval", "completed")


def step_quantize(cfg: JobConfig, manifest: dict) -> None:
    set_step_status(manifest, "quantize", "running")
    script = engine_script(EngineScripts.QUANTIZE)
    args = [
        str(cfg.workspace_dir),
        cfg.onnx_name,
        str(cfg.model_pt),
        "--cali-dir",
        str(cfg.cali_dir),
        "--preprocess-mode",
        cfg.preprocess_mode,
    ]
    r = run_script(script, args, log_path=_log(cfg, "quantize"))
    if r.returncode != 0:
        set_step_status(manifest, "quantize", "failed", message=r.stderr[-2000:] if r.stderr else "")
        raise RuntimeError(f"quantize 失败: exit {r.returncode}")
    set_step_status(manifest, "quantize", "completed", extra={"output_onnx": str(cfg.output_onnx())})


def step_onnx_eval(cfg: JobConfig, manifest: dict, *, input_mode: str = "direct") -> None:
    step_key = "onnx_eval" if input_mode == "direct" else "fpga_eval"
    set_step_status(manifest, step_key, "running")
    script = platform_script(PlatformScripts.ONNX_EVAL)
    args = [
        "--job-dir",
        str(cfg.job_root),
        "--model",
        str(cfg.quantized_onnx()),
        "--input-mode",
        input_mode,
        "--batch-size",
        "8",
    ]
    r = run_script(script, args, log_path=_log(cfg, step_key))
    if r.returncode != 0:
        set_step_status(manifest, step_key, "failed", message=r.stderr[-2000:] if r.stderr else "")
        raise RuntimeError(f"{step_key} 失败: exit {r.returncode}")
    set_step_status(manifest, step_key, "completed")


def step_fpga_test_pack(cfg: JobConfig, manifest: dict) -> None:
    set_step_status(manifest, "fpga_test_pack", "running")
    script = platform_script(PlatformScripts.GENERATE_FPGA_TEST_PACK)
    r = run_script(
        script,
        ["--job-dir", str(cfg.job_root)],
        log_path=_log(cfg, "fpga_test_pack"),
    )
    if r.returncode != 0:
        set_step_status(manifest, "fpga_test_pack", "failed", message=r.stderr[-2000:] if r.stderr else "")
        raise RuntimeError(f"fpga_test_pack 失败: exit {r.returncode}")
    set_step_status(manifest, "fpga_test_pack", "completed")


def step_generate_bin(cfg: JobConfig, manifest: dict) -> None:
    set_step_status(manifest, "generate_bin", "running")
    cfg.layer_bin_dir().mkdir(parents=True, exist_ok=True)
    script = engine_script(EngineScripts.GENERATE_LAYER_BIN)
    model = cfg.output_onnx()
    r = run_script(
        script,
        [str(model), str(cfg.layer_bin_dir())],
        log_path=_log(cfg, "generate_bin"),
    )
    if r.returncode != 0:
        set_step_status(manifest, "generate_bin", "failed", message=r.stderr[-2000:] if r.stderr else "")
        raise RuntimeError(f"generate_bin 失败: exit {r.returncode}")
    set_step_status(manifest, "generate_bin", "completed")


def step_merge_bin(cfg: JobConfig, manifest: dict) -> None:
    """与 process_pipeline.MergeBinStep 相同子步骤，路径来自 script_registry。"""
    set_step_status(manifest, "merge_bin", "running")
    source = cfg.layer_bin_dir()
    output = cfg.all_bin_dir()
    output.mkdir(parents=True, exist_ok=True)
    source_name = source.name
    renamed = output.parent / f"renamed_weights_{source_name}"
    renamed.mkdir(parents=True, exist_ok=True)

    steps = [
        (EngineScripts.RENAME_WT, [str(source), str(renamed)]),
        (EngineScripts.MERGE_WT, [str(renamed), str(output)]),
        (EngineScripts.RENAME_BN, [str(source), str(renamed)]),
        (EngineScripts.MERGE_BN, [str(renamed), str(output)]),
    ]
    for eng, args in steps:
        r = run_script(engine_script(eng), args, log_path=_log(cfg, "merge_bin"))
        if r.returncode != 0:
            set_step_status(manifest, "merge_bin", "failed", message=r.stderr[-2000:] if r.stderr else "")
            raise RuntimeError(f"merge_bin ({eng}) 失败: exit {r.returncode}")
        if eng == EngineScripts.MERGE_WT:
            wt_lower = output / "ALL_wt.bin"
            wt_upper = output / "ALL_WT.bin"
            if wt_lower.is_file() and not wt_upper.is_file():
                wt_lower.replace(wt_upper)
            if not wt_upper.is_file():
                msg = "merge_bin 失败: 缺少 ALL_WT.bin"
                set_step_status(manifest, "merge_bin", "failed", message=msg)
                raise RuntimeError(msg)
    set_step_status(manifest, "merge_bin", "completed")


def step_bundle(cfg: JobConfig, manifest: dict) -> None:
    set_step_status(manifest, "bundle", "running")
    script = platform_script(PlatformScripts.BUNDLE)
    r = run_script(
        script,
        ["--job-dir", str(cfg.job_root)],
        log_path=_log(cfg, "bundle"),
    )
    if r.returncode != 0:
        set_step_status(manifest, "bundle", "failed", message=r.stderr[-2000:] if r.stderr else "")
        raise RuntimeError(f"bundle 失败: exit {r.returncode}")
    set_step_status(manifest, "bundle", "completed", extra={"zip": str(cfg.bundle_zip_path())})


def _step_completed(manifest: dict, step: str) -> bool:
    return (manifest.get("steps") or {}).get(step, {}).get("status") == "completed"


def run_full_pipeline(cfg: JobConfig) -> None:
    scratch = ensure_scratch_ready(cfg)
    cfg.ensure_layout()
    cfg.save()
    write_input_manifest(cfg)
    manifest = load_manifest(cfg.manifest_path)
    manifest["display_name"] = cfg.display_name
    manifest["job_id"] = cfg.job_id
    manifest["status"] = "running"
    manifest["scratch"] = scratch
    manifest.pop("error", None)
    save_manifest(cfg.manifest_path, manifest)

    ordered = [
        ("pt_eval", lambda: step_pt_eval(cfg, manifest)),
        ("quantize", lambda: step_quantize(cfg, manifest)),
        ("onnx_eval", lambda: step_onnx_eval(cfg, manifest, input_mode="direct")),
        ("fpga_test_pack", lambda: step_fpga_test_pack(cfg, manifest)),
        ("fpga_eval", lambda: step_onnx_eval(cfg, manifest, input_mode="fpga_side_view")),
        ("generate_bin", lambda: step_generate_bin(cfg, manifest)),
        ("merge_bin", lambda: step_merge_bin(cfg, manifest)),
        ("bundle", lambda: step_bundle(cfg, manifest)),
    ]

    try:
        for name, fn in ordered:
            if _step_completed(manifest, name):
                print(f"skip {name} (already completed)")
                continue
            fn()
            save_manifest(cfg.manifest_path, manifest)
        manifest["status"] = "completed"
        save_manifest(cfg.manifest_path, manifest)
        try:
            manifest["scratch_cleanup"] = finalize_success_scratch(cfg)
        except Exception as cleanup_error:
            # 成果已在持久盘完成；清理失败只进入重试状态，不反转任务结果。
            manifest["scratch_cleanup"] = {
                "status": "pending",
                "error": str(cleanup_error),
            }
    except Exception as e:
        manifest["status"] = "failed"
        manifest["error"] = str(e)
        manifest["failure_archive_status"] = "copying" if cfg.use_scratch else "not_applicable"
        save_manifest(cfg.manifest_path, manifest)
        try:
            archive = archive_failed_scratch(cfg)
            manifest["failure_archive_status"] = archive.get("status", "verified")
            manifest["failure_archive"] = {
                key: archive.get(key)
                for key in ("archive", "file_count", "total_bytes", "verified_at_epoch")
                if archive.get(key) is not None
            }
        except Exception as archive_error:
            manifest["failure_archive_status"] = "pending"
            manifest["failure_archive_error"] = str(archive_error)
        save_manifest(cfg.manifest_path, manifest)
        raise
    save_manifest(cfg.manifest_path, manifest)
