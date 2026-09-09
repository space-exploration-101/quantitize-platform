#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将 output_data/ 旧 job 打成 tar 归档到 902（或其它根目录）。

仅处理 QUANTITIZE_DIR/output_data 下的子目录；不触碰 shared_data 等。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from job_config import OUTPUT_DATA_ROOT, QUANTITIZE_DIR

# 正式 job：YYYYMMDD_HHMMSS_name
_DATED_JOB_RE = re.compile(r"^(\d{8})_(\d{6})_.+")

# 默认识别的 902 归档根（可被 --archive-root / 环境变量覆盖）
_DEFAULT_ARCHIVE_CANDIDATES = (
    Path("/mnt/902/3-个人/33-王睿索老师/fpga_zip"),
    Path(
        "/run/user/1000/gvfs/smb-share:server=orange.local,share=902_data"
        "/3-个人/33-王睿索老师/fpga_zip"
    ),
    Path(
        "/run/user/1000/gvfs/smb-share:server=10.2.29.106,share=902_data"
        "/3-个人/33-王睿索老师/fpga_zip"
    ),
)

_DEFAULT_STAGING_CANDIDATES = (
    Path("/media/rs/新加卷1/fpga_zip_staging"),
    Path("/tmp/fpga_zip_staging"),
)

ENV_ARCHIVE_ROOT = "FPGA_ZIP_ARCHIVE_ROOT"
ENV_STAGING_ROOT = "FPGA_ZIP_STAGING_ROOT"


@dataclass
class JobEntry:
    name: str
    path: Path
    dated: bool
    sort_key: str  # YYYYMMDDHHMMSS or name
    bytes: int


@dataclass
class ArchiveResult:
    job_id: str
    status: str  # archived | kept | skipped | dry_run | error
    message: str = ""
    archive_path: str = ""
    source_bytes: int = 0
    archive_bytes: int = 0


def resolve_archive_root(explicit: Optional[Path] = None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    env = os.environ.get(ENV_ARCHIVE_ROOT, "").strip()
    if env:
        return Path(env).expanduser().resolve()
    for cand in _DEFAULT_ARCHIVE_CANDIDATES:
        parent = cand.parent
        if parent.is_dir():
            return cand
    raise FileNotFoundError(
        "找不到 902 归档根目录。请挂载 //10.2.29.106/902_data，"
        f"或设置 {ENV_ARCHIVE_ROOT} / --archive-root。"
    )


def resolve_staging_root(explicit: Optional[Path] = None) -> Path:
    if explicit is not None:
        p = explicit.expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p
    env = os.environ.get(ENV_STAGING_ROOT, "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p
    for cand in _DEFAULT_STAGING_CANDIDATES:
        try:
            parent = cand.parent
            if parent.is_dir():
                cand.mkdir(parents=True, exist_ok=True)
                return cand
        except OSError:
            continue
    p = Path("/tmp/fpga_zip_staging")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _dir_bytes(path: Path) -> int:
    """目录占用（不跟随符号链接；与 tar 默认行为一致）。"""
    total = 0
    for root, _dirs, files in os.walk(path, followlinks=False):
        for name in files:
            fp = Path(root) / name
            try:
                st = fp.lstat()
                if not Path(fp).is_symlink():
                    total += st.st_size
            except OSError:
                pass
    return total


def list_output_jobs(output_root: Path = OUTPUT_DATA_ROOT) -> List[JobEntry]:
    if not output_root.is_dir():
        return []
    entries: List[JobEntry] = []
    for child in sorted(output_root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        m = _DATED_JOB_RE.match(child.name)
        dated = m is not None
        sort_key = f"{m.group(1)}{m.group(2)}" if m else child.name
        entries.append(
            JobEntry(
                name=child.name,
                path=child,
                dated=dated,
                sort_key=sort_key,
                bytes=_dir_bytes(child),
            )
        )
    return entries


def plan_keep_and_archive(
    jobs: Sequence[JobEntry],
    *,
    keep: int = 2,
    archive_aux: bool = True,
) -> Tuple[List[JobEntry], List[JobEntry]]:
    """返回 (keep_list, archive_list)。

    - 正式 dated job：按时间戳降序，保留最新 keep 个。
    - 非 dated（smoke/regression 等）：archive_aux=True 时全部归档。
    """
    dated = sorted([j for j in jobs if j.dated], key=lambda j: j.sort_key, reverse=True)
    aux = [j for j in jobs if not j.dated]
    keep_list = list(dated[: max(0, keep)])
    archive_list = list(dated[max(0, keep) :])
    if archive_aux:
        archive_list.extend(sorted(aux, key=lambda j: j.name))
    else:
        keep_list.extend(aux)
    return keep_list, archive_list


def _write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _update_index(archive_root: Path, entry: dict) -> None:
    index_path = archive_root / "index.json"
    items: List[dict] = []
    if index_path.is_file():
        try:
            raw = json.loads(index_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and isinstance(raw.get("archives"), list):
                items = list(raw["archives"])
            elif isinstance(raw, list):
                items = raw
        except json.JSONDecodeError:
            items = []
    items = [x for x in items if x.get("job_id") != entry.get("job_id")]
    items.append(entry)
    items.sort(key=lambda x: x.get("job_id", ""))
    _write_manifest(
        index_path,
        {
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "archives": items,
        },
    )


def archive_one_job(
    job: JobEntry,
    *,
    archive_root: Path,
    staging_root: Path,
    output_root: Path = OUTPUT_DATA_ROOT,
    dry_run: bool = False,
    delete_local: bool = True,
) -> ArchiveResult:
    """本地 tar → 拷到 archive_root → 校验大小 → 删本地 job。"""
    job_id = job.name
    if not job.path.is_dir():
        return ArchiveResult(job_id, "skipped", "目录不存在")

    if job.path.resolve().parent != output_root.resolve():
        return ArchiveResult(job_id, "error", "拒绝：job 不在 output_data/ 下")

    tar_name = f"{job_id}.tar"
    staging_tar = staging_root / tar_name
    dest_tar = archive_root / tar_name
    dest_meta = archive_root / f"{job_id}.manifest.json"
    source_bytes = job.bytes or _dir_bytes(job.path)

    if dry_run:
        return ArchiveResult(
            job_id,
            "dry_run",
            f"将归档 {source_bytes} bytes → {dest_tar}",
            archive_path=str(dest_tar),
            source_bytes=source_bytes,
        )

    archive_root.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir(parents=True, exist_ok=True)

    try:
        if not dest_tar.is_file():
            if staging_tar.is_file():
                staging_tar.unlink()
            # tar 直接写到有空间的 staging，避免占满系统盘
            subprocess.run(
                ["tar", "-cf", str(staging_tar), "-C", str(output_root), job_id],
                check=True,
            )
            shutil.copy2(staging_tar, dest_tar)

        archive_bytes = dest_tar.stat().st_size
        if staging_tar.is_file():
            local_bytes = staging_tar.stat().st_size
            if local_bytes != archive_bytes:
                return ArchiveResult(
                    job_id,
                    "error",
                    f"大小不一致 staging={local_bytes} remote={archive_bytes}",
                    archive_path=str(dest_tar),
                    source_bytes=source_bytes,
                    archive_bytes=archive_bytes,
                )

        meta = {
            "job_id": job_id,
            "archived_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "source_bytes": source_bytes,
            "archive_bytes": archive_bytes,
            "archive_name": tar_name,
            "source_path": f"output_data/{job_id}",
            "quantitize_dir": str(QUANTITIZE_DIR),
        }
        _write_manifest(dest_meta, meta)
        _update_index(archive_root, meta)

        if delete_local:
            shutil.rmtree(job.path)
        if staging_tar.is_file():
            staging_tar.unlink()

        return ArchiveResult(
            job_id,
            "archived",
            "ok",
            archive_path=str(dest_tar),
            source_bytes=source_bytes,
            archive_bytes=archive_bytes,
        )
    except Exception as exc:  # noqa: BLE001 — CLI 汇总错误
        return ArchiveResult(job_id, "error", str(exc), source_bytes=source_bytes)


def run_archive(
    *,
    keep: int = 2,
    archive_aux: bool = True,
    dry_run: bool = False,
    delete_local: bool = True,
    output_root: Path = OUTPUT_DATA_ROOT,
    archive_root: Optional[Path] = None,
    staging_root: Optional[Path] = None,
    only: Optional[Iterable[str]] = None,
) -> List[ArchiveResult]:
    output_root = output_root.resolve()
    if output_root.name != "output_data":
        raise ValueError(f"安全限制：output_root 必须名为 output_data，收到: {output_root}")

    jobs = list_output_jobs(output_root)
    keep_list, archive_list = plan_keep_and_archive(jobs, keep=keep, archive_aux=archive_aux)

    if only is not None:
        only_set = set(only)
        archive_list = [j for j in archive_list if j.name in only_set]
        # 显式 only 时，不在 only 里的 keep 仍标记 kept
        keep_list = [j for j in keep_list if j.name not in only_set]

    arch_root = resolve_archive_root(archive_root)
    stag_root = resolve_staging_root(staging_root)

    results: List[ArchiveResult] = []
    for j in keep_list:
        results.append(
            ArchiveResult(
                j.name,
                "kept",
                f"保留本机 ({j.bytes} bytes)",
                source_bytes=j.bytes,
            )
        )
    for j in archive_list:
        results.append(
            archive_one_job(
                j,
                archive_root=arch_root,
                staging_root=stag_root,
                output_root=output_root,
                dry_run=dry_run,
                delete_local=delete_local,
            )
        )
    return results


def format_plan(results: Sequence[ArchiveResult]) -> str:
    lines = []
    for r in results:
        extra = f"  {r.message}" if r.message else ""
        lines.append(f"[{r.status:8}] {r.job_id}{extra}")
    return "\n".join(lines)
