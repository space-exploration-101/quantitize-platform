#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""初始化 shared_data/ 默认标定集与测试集。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_LIB = Path(__file__).resolve().parent / "_lib"
if str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))
from bootstrap import bootstrap_platform_script  # noqa: E402

bootstrap_platform_script()
from shared_datasets import (  # noqa: E402
    DEFAULT_TEST_DATASET_ID,
    SHARED_DATA_ROOT,
    bootstrap_default_datasets,
    list_cali_datasets,
    list_dataset_catalog,
    list_test_datasets,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="初始化共享数据集")
    parser.add_argument("--force", action="store_true", help="强制重建标定链接 / 测试图 1280")
    parser.add_argument(
        "--resize-test-only",
        action="store_true",
        help="仅将共享测试集图像物化为 1280×1280（不改 cali、不碰 Quantitative test 原图）",
    )
    args = parser.parse_args()
    if args.resize_test_only:
        from shared_datasets import (  # noqa: WPS433
            SOURCE_TEST_IMG,
            SOURCE_TEST_LAB,
            SHARED_TEST_IMGSZ,
            materialize_shared_test_images,
            test_root,
        )

        dest = test_root(DEFAULT_TEST_DATASET_ID)
        n = materialize_shared_test_images(dest, SOURCE_TEST_IMG, SOURCE_TEST_LAB)
        bootstrap_default_datasets(force=False)
        print(f"已物化 {n} 张测试图 → {dest / 'images'} ({SHARED_TEST_IMGSZ}×{SHARED_TEST_IMGSZ})")
    else:
        bootstrap_default_datasets(force=args.force)
    print(f"共享数据根目录: {SHARED_DATA_ROOT}")
    print("标定集:")
    for d in list_cali_datasets():
        print(f"  [{d.id}] {d.display_name} — {d.image_count} 张 @ {d.root}")
    print("数据集目录（测试已拆分为原始图与 FPGA 包）:")
    for row in list_dataset_catalog():
        if row.download_kind == "cali":
            continue
        d = row.entry
        print(f"  [{row.kind_label}] {d.id} — {d.display_name} — {d.image_count or '—'} 张 @ {d.rel_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
