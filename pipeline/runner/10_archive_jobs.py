#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""步骤：归档 output_data/ 旧任务到 902 fpga_zip。

本机默认只保留最新 2 个带时间戳的正式 job；更早的（及 smoke/regression）
打成 .tar 后拷到归档根，校验大小后再删除本机目录。

示例：
  python platform/10_archive_jobs.py --dry-run
  python platform/10_archive_jobs.py --keep 2
  python platform/10_archive_jobs.py --only 20260715_122542_aituosh_moon_final_13
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_LIB = Path(__file__).resolve().parent / "_lib"
if str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))
from bootstrap import bootstrap_platform_script  # noqa: E402

bootstrap_platform_script()
from archive_jobs import (  # noqa: E402
    ENV_ARCHIVE_ROOT,
    ENV_STAGING_ROOT,
    OUTPUT_DATA_ROOT,
    format_plan,
    resolve_archive_root,
    resolve_staging_root,
    run_archive,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="归档 output_data/ 旧 job 到 902 fpga_zip")
    parser.add_argument("--keep", type=int, default=2, help="本机保留最新正式 job 个数（默认 2）")
    parser.add_argument(
        "--no-archive-aux",
        action="store_true",
        help="不归档 smoke/regression 等非时间戳目录（默认会归档）",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不打包不删除")
    parser.add_argument(
        "--keep-local",
        action="store_true",
        help="上传成功后仍保留本机目录（默认删除）",
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=None,
        help=f"902 归档目录（或设 {ENV_ARCHIVE_ROOT}）",
    )
    parser.add_argument(
        "--staging-root",
        type=Path,
        default=None,
        help=f"本机临时 tar 目录（或设 {ENV_STAGING_ROOT}）",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_DATA_ROOT,
        help="仅允许指向 …/output_data",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="只归档指定 job 名（仍须在归档候选里）",
    )
    args = parser.parse_args()

    try:
        arch = resolve_archive_root(args.archive_root)
        stag = resolve_staging_root(args.staging_root)
    except FileNotFoundError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    print(f"output_root : {args.output_root}")
    print(f"archive_root: {arch}")
    print(f"staging_root: {stag}")
    print(f"keep={args.keep} dry_run={args.dry_run} archive_aux={not args.no_archive_aux}")

    results = run_archive(
        keep=args.keep,
        archive_aux=not args.no_archive_aux,
        dry_run=args.dry_run,
        delete_local=not args.keep_local,
        output_root=args.output_root,
        archive_root=arch,
        staging_root=stag,
        only=args.only,
    )
    print(format_plan(results))

    errors = [r for r in results if r.status == "error"]
    archived = [r for r in results if r.status == "archived"]
    print(f"\n汇总: archived={len(archived)} errors={len(errors)} total={len(results)}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
