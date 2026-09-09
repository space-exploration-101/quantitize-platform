#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""步骤 ⑧：按 WEB_DESIGN.md §6.2 打包下载 zip。"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

_LIB = Path(__file__).resolve().parent / "_lib"
if str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))
from bootstrap import bootstrap_platform_script  # noqa: E402

bootstrap_platform_script()
from bundle_inventory import arcname_in_bundle, inventory_job_dir, inventory_zip  # noqa: E402
from job_config import JobConfig  # noqa: E402


def _add_dir(
    zf: zipfile.ZipFile,
    base: Path,
    arc_prefix: str,
    *,
    onnx_name: str,
    slim: bool,
) -> None:
    if not base.is_dir():
        return
    for path in base.rglob("*"):
        if path.is_file():
            arc = str(Path(arc_prefix) / path.relative_to(base))
            if not arcname_in_bundle(arc, onnx_name, slim=slim):
                continue
            zf.write(path, arcname=arc)


def main() -> int:
    parser = argparse.ArgumentParser(description="打包量化产物 zip")
    parser.add_argument("--job-dir", required=True, type=Path)
    parser.add_argument(
        "--slim",
        action="store_true",
        default=True,
        help="精简包：排除 workspace/bin 层间调试 bin（默认开启）",
    )
    parser.add_argument("--full", action="store_true", help="完整包（仍不含中间 ONNX，见 WEB_DESIGN §6.2）")
    args = parser.parse_args()
    slim = not args.full
    cfg = JobConfig.from_job_dir(args.job_dir)
    zip_path = cfg.bundle_zip_path()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        _add_dir(zf, cfg.workspace_dir, "workspace", onnx_name=cfg.onnx_name, slim=slim)
        _add_dir(zf, cfg.results_dir, "results", onnx_name=cfg.onnx_name, slim=slim)
        _add_dir(zf, cfg.logs_dir, "logs", onnx_name=cfg.onnx_name, slim=slim)
        if cfg.manifest_path.is_file():
            zf.write(cfg.manifest_path, arcname="manifest.json")
        readme = cfg.job_root / "README_bundle.md"
        mode = "精简" if slim else "标准"
        readme.write_text(
            f"# 量化产物包（{mode}）\n\n"
            "见 WEB_DESIGN.md §6.2。\n"
            "不含 input/model.pt、共享标定/测试集原始图、fpga_test_pack/。\n"
            f"\n成果 workspace ONNX 仅含 `{cfg.onnx_name}_output.onnx`（主推理模型）。\n"
            f"不含：`{cfg.onnx_name}.onnx`、`*_fp16*.onnx`（量化中间文件，仍在任务 workspace/ 内）。\n"
            + (
                "\n含 `workspace/renamed_weights_bin/` 每层 01_rename wt + 03_rename bn，"
                "及 `workspace/all_bin/` 合并权重。\n"
                "不含 `workspace/bin/` 原始层调试 bin。\n"
                if slim
                else ""
            ),
            encoding="utf-8",
        )
        zf.write(readme, arcname="README_bundle.md")

    inv = inventory_zip(zip_path)
    inv_path = cfg.results_dir / "bundle_inventory.json"
    inv_path.parent.mkdir(parents=True, exist_ok=True)
    inv_path.write_text(json.dumps(inv, indent=2), encoding="utf-8")
    dir_inv = inventory_job_dir(
        cfg.job_root,
        onnx_name=cfg.onnx_name,
        deliverable_only=True,
        workspace_dir=cfg.workspace_dir,
        fpga_test_pack_dir=cfg.fpga_test_pack_dir,
    )
    (cfg.results_dir / "job_dir_inventory.json").write_text(
        json.dumps(dir_inv, indent=2), encoding="utf-8"
    )

    print(f"已打包: {zip_path} ({inv['total_bytes'] / 1e6:.0f} MB 解压后)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
