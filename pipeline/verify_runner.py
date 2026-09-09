#!/usr/bin/env python3
"""M3 冒烟：registry + bootstrap + JobConfig + validate 模块可导入。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "pipeline" / "runner"
LIB = RUNNER / "_lib"
sys.path[:0] = [str(LIB), str(ROOT / "pipeline"), str(ROOT / "pipeline" / "engine")]


def main() -> int:
    from script_registry import (
        QUANTITIZE_LEGACY_DIR,
        all_registered_scripts,
        OUTPUT_DATA_ROOT,
        SHARED_DATA_ROOT,
    )

    scripts = all_registered_scripts()
    print(f"registered={len(scripts)}")
    for k in sorted(scripts):
        p = Path(scripts[k])
        if not p.is_file():
            print(f"MISSING {k} -> {p}")
            return 1

    from bootstrap import bootstrap_platform_script

    root = bootstrap_platform_script()
    print(f"PLATFORM_ROOT={root}")
    print(f"OUTPUT_DATA_ROOT={OUTPUT_DATA_ROOT}")
    print(f"SHARED_DATA_ROOT={SHARED_DATA_ROOT}")
    print(f"QUANTITIZE_LEGACY_DIR={QUANTITIZE_LEGACY_DIR}")

    from job_config import JobConfig, OUTPUT_DATA_ROOT as JC_OUT
    from validate_input import validate_job_input  # noqa: F401
    from runner_core import run_full_pipeline  # noqa: F401
    from env import resolve_yolov8_python, clean_pythonpath
    from manifest import load_manifest, save_manifest
    import os

    py = resolve_yolov8_python()
    print(f"yolov8_python={py}")
    env = clean_pythonpath(os.environ.copy())
    print(f"PYTHONPATH={env.get('PYTHONPATH')}")
    assert "quantitize-platform/pipeline/runner/_lib" in env["PYTHONPATH"].replace("\\", "/")
    assert Path(JC_OUT).resolve() == Path(OUTPUT_DATA_ROOT).resolve()
    assert Path(OUTPUT_DATA_ROOT).is_absolute()
    assert Path(SHARED_DATA_ROOT).is_absolute()
    legacy_output = str(Path(QUANTITIZE_LEGACY_DIR) / "output_data")
    assert legacy_output not in str(OUTPUT_DATA_ROOT)

    cfg = JobConfig.create_task("p1t2_path_smoke", onnx_name="smoke", task_id="_p1t2_path_smoke")
    manifest = load_manifest(cfg.manifest_path)
    manifest["job_id"] = cfg.job_id
    manifest["job_root"] = str(cfg.job_root)
    save_manifest(cfg.manifest_path, manifest)
    blob = cfg.manifest_path.read_text(encoding="utf-8")
    blob += "\n" + (cfg.job_root / "job_config.json").read_text(encoding="utf-8")
    if legacy_output in blob or "/quantitize/output_data" in blob:
        print(f"FAIL: smoke job contains legacy path {legacy_output}")
        return 1
    print(f"SMOKE_JOB={cfg.job_root}")
    print("PASS: runner smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
