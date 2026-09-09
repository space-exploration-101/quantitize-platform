#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""流水线各步断言（TEST_PLAN.md §4）。"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List

import onnx

_LIB = Path(__file__).resolve().parents[1] / "_lib"
if str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))

from job_config import JobConfig

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _fail(checks: List[Dict[str, Any]], step: str, msg: str) -> None:
    checks.append({"step": step, "ok": False, "message": msg})


def _ok(checks: List[Dict[str, Any]], step: str, msg: str = "ok") -> None:
    checks.append({"step": step, "ok": True, "message": msg})


def check_input(cfg: JobConfig, checks: List[Dict[str, Any]]) -> None:
    if cfg.input_manifest_path.is_file():
        _ok(checks, "0_manifest", "input_manifest.md exists")
    else:
        _fail(checks, "0_manifest", "missing input_manifest.md")
    val = cfg.input_validation_path
    if val.is_file():
        data = json.loads(val.read_text())
        if data.get("ok"):
            _ok(checks, "0_validate", "validation ok")
        else:
            _fail(checks, "0_validate", f"validation failed: {data.get('errors')}")
    else:
        _fail(checks, "0_validate", "missing input_validation.json")


def check_eval_summary(summary_path: Path, checks: List[Dict[str, Any]], step: str) -> None:
    if not summary_path.is_file():
        _fail(checks, step, f"missing {summary_path}")
        return
    s = json.loads(summary_path.read_text())
    for key in ("precision", "recall", "f1", "mAP", "pixel_error"):
        if key not in s:
            _fail(checks, step, f"missing key {key}")
            return
    _ok(checks, step, f"metrics ok n_images={s.get('n_images')}")


def check_quantize(cfg: JobConfig, checks: List[Dict[str, Any]]) -> None:
    ws = cfg.workspace_dir
    for name in (cfg.fp16_onnx(), cfg.quantized_onnx(), cfg.output_onnx()):
        if not name.is_file():
            _fail(checks, "2_quantize", f"missing {name.name}")
            return
    try:
        onnx.load(str(cfg.output_onnx()))
    except Exception as e:
        _fail(checks, "2_quantize", f"onnx load failed: {e}")
        return
    size_mb = cfg.output_onnx().stat().st_size / (1024 * 1024)
    if size_mb < 50:
        _fail(checks, "2_quantize", f"onnx too small: {size_mb:.1f}MB")
    else:
        _ok(checks, "2_quantize", f"onnx ~{size_mb:.0f}MB")


def check_fpga_pack(cfg: JobConfig, checks: List[Dict[str, Any]]) -> None:
    pack = cfg.fpga_test_pack_dir
    n_test = len([p for p in cfg.test_images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    bins = list((pack / "bins").glob("*.bin"))
    sides = list((pack / "side_view").glob("*.png"))
    if len(bins) != n_test or len(sides) != n_test:
        _fail(checks, "4_fpga_pack", f"count mismatch bins={len(bins)} side={len(sides)} test={n_test}")
        return
    eval_py = pack / "scripts" / "eval_fpga_results.py"
    if not eval_py.is_file():
        _fail(checks, "4_fpga_pack", "missing eval_fpga_results.py")
        return
    others = [p for p in (pack / "scripts").iterdir() if p.name != "eval_fpga_results.py"]
    if others:
        _fail(checks, "4_fpga_pack", f"extra scripts: {[p.name for p in others]}")
    elif not (pack / "labels_1280.json").is_file():
        _fail(checks, "4_fpga_pack", "missing labels_1280.json")
    else:
        _ok(checks, "4_fpga_pack", f"{len(bins)} bins + eval script")


def check_generate_bin(cfg: JobConfig, checks: List[Dict[str, Any]]) -> None:
    bdir = cfg.layer_bin_dir()
    if not bdir.is_dir() or not any(bdir.iterdir()):
        _fail(checks, "6_generate_bin", "bin/ empty")
    else:
        subdirs = [d for d in bdir.iterdir() if d.is_dir()]
        _ok(checks, "6_generate_bin", f"{len(subdirs)} layer dirs")


def check_merge_bin(cfg: JobConfig, checks: List[Dict[str, Any]]) -> None:
    wt = cfg.all_bin_dir() / "ALL_WT.bin"
    if not wt.is_file():
        wt = cfg.all_bin_dir() / "ALL_wt.bin"
    bn = cfg.all_bin_dir() / "ALL_BN.bin"
    if not wt.is_file() or not bn.is_file():
        _fail(checks, "7_merge_bin", "ALL_WT/BN missing")
        return
    wt_mb = wt.stat().st_size / (1024 * 1024)
    if wt_mb < 50:
        _fail(checks, "7_merge_bin", f"ALL_WT too small {wt_mb:.1f}MB")
    else:
        _ok(checks, "7_merge_bin", f"WT {wt_mb:.0f}MB BN {bn.stat().st_size // 1024}KB")


def check_bundle(cfg: JobConfig, checks: List[Dict[str, Any]]) -> None:
    zp = cfg.bundle_zip_path()
    if not zp.is_file():
        _fail(checks, "8_bundle", "zip missing")
        return
    with zipfile.ZipFile(zp, "r") as zf:
        names = zf.namelist()
        if any("input/model.pt" in n for n in names):
            _fail(checks, "8_bundle", "zip contains input/model.pt")
            return
        for prefix in ("workspace/", "results/", "logs/"):
            if not any(n.startswith(prefix) for n in names):
                _fail(checks, "8_bundle", f"zip missing {prefix}")
                return
        if any(n.startswith("fpga_test_pack/") for n in names):
            _fail(checks, "8_bundle", "zip should not contain fpga_test_pack/")
            return
        if not any("renamed_weights_" in n and n.endswith("_wt.bin") for n in names):
            _fail(checks, "8_bundle", "zip missing renamed layer wt bins")
            return
        mid_onnx = f"workspace/{cfg.onnx_name}.onnx"
        if mid_onnx in names:
            _fail(checks, "8_bundle", f"zip should not contain {mid_onnx}")
            return
        if not any(n.endswith("_output.onnx") for n in names):
            _fail(checks, "8_bundle", "zip missing *_output.onnx")
            return
    _ok(checks, "8_bundle", str(zp.name))


def run_all_checks(cfg: JobConfig, *, skip_quantize_heavy: bool = False) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    check_input(cfg, checks)
    check_eval_summary(cfg.pt_eval_dir() / "summary.json", checks, "1_pt_eval")
    if not skip_quantize_heavy:
        check_quantize(cfg, checks)
    check_eval_summary(cfg.onnx_eval_dir() / "summary.json", checks, "3_onnx_eval")
    check_fpga_pack(cfg, checks)
    check_eval_summary(cfg.fpga_eval_dir() / "summary.json", checks, "5_fpga_eval")
    check_generate_bin(cfg, checks)
    check_merge_bin(cfg, checks)
    check_bundle(cfg, checks)
    passed = all(c["ok"] for c in checks)
    return {"ok": passed, "checks": checks}


def print_report(report: Dict[str, Any]) -> None:
    for c in report["checks"]:
        mark = "PASS" if c["ok"] else "FAIL"
        print(f"[{mark}] {c['step']}: {c['message']}")
    print(f"\nOverall: {'PASS' if report['ok'] else 'FAIL'}")
