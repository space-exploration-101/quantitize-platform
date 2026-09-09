#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""创建任务：ZIP 上传与共享数据集（从旧 Web 迁入，返回 JSON）。"""

from __future__ import annotations

from dataclasses import asdict
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any, Optional

from fastapi import HTTPException, UploadFile

from apps.api._paths import setup_pipeline_imports

setup_pipeline_imports()

from input_manifest import write_input_manifest  # noqa: E402
from job_config import JobConfig  # noqa: E402
from manifest import load_manifest, save_manifest  # noqa: E402
from preprocess import InputPreprocessMode, normalize_preprocess_mode  # noqa: E402
from shared_datasets import apply_test_dataset_nc, attach_datasets_to_job  # noqa: E402
from validate_input import save_report, validate_job_input  # noqa: E402

MAX_ZIP_BYTES = 5 * 1024 * 1024 * 1024
DEFAULT_PREPROCESS = InputPreprocessMode.GRAYSCALE_UNIFORM.value


def _http_400(detail: str) -> HTTPException:
    return HTTPException(status_code=400, detail=detail)


def _init_manifest(cfg: JobConfig) -> None:
    manifest = load_manifest(cfg.manifest_path)
    manifest["job_id"] = cfg.job_id
    manifest["job_root"] = str(cfg.job_root)
    save_manifest(cfg.manifest_path, manifest)


def _cleanup_job(cfg: JobConfig) -> None:
    shutil.rmtree(cfg.job_root, ignore_errors=True)


def _safe_zip_target(dest: Path, member: str) -> Path:
    dest = dest.resolve()
    if member.startswith("/") or member.startswith("\\") or Path(member).is_absolute():
        raise _http_400(f"非法 zip 路径: {member}")
    # zip 内不得含盘符或父目录穿越
    parts = Path(member.replace("\\", "/")).parts
    if any(p in ("..", "") for p in parts if p != "/"):
        if ".." in parts:
            raise _http_400(f"非法 zip 路径: {member}")
    target = (dest / member).resolve()
    try:
        target.relative_to(dest)
    except ValueError:
        raise _http_400(f"非法 zip 路径: {member}")
    return target


def safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    uncompressed = sum(info.file_size for info in zf.infolist())
    if uncompressed > MAX_ZIP_BYTES:
        raise _http_400("zip 解压后超过 5GB 限制")
    for member in zf.namelist():
        if member.endswith("/") or member.endswith("\\"):
            continue
        target = _safe_zip_target(dest, member)
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(member) as src, open(target, "wb") as out:
            shutil.copyfileobj(src, out)


def find_input_root(extract_dir: Path) -> Path:
    if (extract_dir / "model.pt").is_file():
        return extract_dir
    if (extract_dir / "input" / "model.pt").is_file():
        return extract_dir / "input"
    for child in sorted(p for p in extract_dir.iterdir() if p.is_dir()):
        if (child / "model.pt").is_file():
            return child
    raise _http_400("zip 内未找到 model.pt（根目录或 input/ 下）")


def _task_payload(cfg: JobConfig, *, validation_ok: bool, errors: list | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "ok": validation_ok,
        "task_id": cfg.job_id,
        "display_name": cfg.display_name,
        "status": "pending",
        "job_root": str(cfg.job_root),
        "validation_ok": validation_ok,
    }
    if errors:
        body["errors"] = errors
    return body


def create_from_zip(
    *,
    display_name: str,
    onnx_name: str,
    preprocess_mode: str,
    content: bytes,
) -> tuple[int, dict[str, Any]]:
    if len(content) > MAX_ZIP_BYTES:
        raise _http_400("zip 超过 5GB 限制")
    name = display_name.strip() or "zip_task"
    try:
        mode = normalize_preprocess_mode(preprocess_mode)
    except ValueError as e:
        raise _http_400(str(e)) from e
    oname = onnx_name.strip() or name
    cfg = JobConfig.create_task(name, onnx_name=oname, preprocess_mode=mode)
    tmp_zip = cfg.job_root / "_upload.zip"
    extract_tmp = cfg.job_root / "_extract"
    try:
        tmp_zip.write_bytes(content)
        if not zipfile.is_zipfile(tmp_zip):
            raise _http_400("不是有效的 zip 文件")
        if extract_tmp.exists():
            shutil.rmtree(extract_tmp)
        extract_tmp.mkdir()
        with zipfile.ZipFile(tmp_zip, "r") as zf:
            safe_extract(zf, extract_tmp)
        src_root = find_input_root(extract_tmp)
        inp = cfg.input_dir
        inp.mkdir(parents=True, exist_ok=True)
        for item in ("model.pt", "cali", "test"):
            src = src_root / item
            dst = inp / item
            if src.is_file():
                shutil.copy2(src, dst)
            elif src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
        jc_src = extract_tmp / "job_config.json"
        if not jc_src.is_file() and (src_root / "job_config.json").is_file():
            jc_src = src_root / "job_config.json"
        if jc_src.is_file():
            try:
                data = json.loads(jc_src.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                raise _http_400(f"job_config.json 无效: {e}") from e
            if not isinstance(data, dict):
                raise _http_400("job_config.json 必须是对象")
            data.pop("job_root", None)
            data.pop("detect_head_exclude_mode", None)
            data["job_id"] = cfg.job_id
            data["display_name"] = name
            if onnx_name.strip():
                data["onnx_name"] = onnx_name.strip()
            data["preprocess_mode"] = mode
            (cfg.job_root / "job_config.json").write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            cfg = JobConfig.from_job_dir(cfg.job_root)
        else:
            cfg.preprocess_mode = mode
        cfg.save()
        _init_manifest(cfg)
        report = validate_job_input(
            cfg.job_root,
            min_cali=cfg.min_cali_images,
            min_test=cfg.min_test_images,
            nc=cfg.nc,
            expect_imgsz=cfg.imgsz,
        )
        save_report(report, cfg.input_validation_path)
        write_input_manifest(cfg, validation_ok=report.ok)
        cfg.save()
        errors = [asdict(e) for e in report.errors]
        payload = _task_payload(cfg, validation_ok=report.ok, errors=errors or None)
        return (201 if report.ok else 400), payload
    except HTTPException:
        _cleanup_job(cfg)
        raise
    except zipfile.BadZipFile as e:
        _cleanup_job(cfg)
        raise _http_400(f"zip 损坏: {e}") from e
    except OSError as e:
        _cleanup_job(cfg)
        raise _http_400(f"解压失败: {e}") from e
    finally:
        shutil.rmtree(extract_tmp, ignore_errors=True)
        tmp_zip.unlink(missing_ok=True)


async def read_upload(upload: Optional[UploadFile], *, limit: int, what: str) -> bytes:
    if upload is None:
        raise _http_400(f"缺少 {what}")
    content = await upload.read()
    if not content:
        raise _http_400(f"{what} 为空")
    if len(content) > limit:
        raise _http_400(f"{what} 超过 5GB 限制")
    return content


def create_from_shared(
    *,
    display_name: str,
    onnx_name: str,
    preprocess_mode: str,
    cali_dataset_id: str,
    test_dataset_id: str,
    imgsz: int,
    model_bytes: bytes,
    min_cali_images: Optional[int] = None,
    min_test_images: Optional[int] = None,
    nc: Optional[int] = None,
) -> tuple[int, dict[str, Any]]:
    name = display_name.strip() or "shared_task"
    cali_id = cali_dataset_id.strip()
    test_id = test_dataset_id.strip()
    if not cali_id or not test_id:
        raise _http_400("需要 cali_dataset_id 和 test_dataset_id")
    try:
        mode = normalize_preprocess_mode(preprocess_mode)
    except ValueError as e:
        raise _http_400(str(e)) from e
    if imgsz <= 0:
        raise _http_400("imgsz 必须为正整数")
    oname = onnx_name.strip() or name
    extra: dict[str, Any] = {
        "preprocess_mode": mode,
        "imgsz": imgsz,
        "cali_dataset_id": cali_id,
        "test_dataset_id": test_id,
    }
    if min_cali_images is not None:
        extra["min_cali_images"] = min_cali_images
    if min_test_images is not None:
        extra["min_test_images"] = min_test_images
    if nc is not None:
        extra["nc"] = nc
    cfg = JobConfig.create_task(name, onnx_name=oname, **extra)
    try:
        pt_path = cfg.model_pt
        pt_path.parent.mkdir(parents=True, exist_ok=True)
        pt_path.write_bytes(model_bytes)
        attach_datasets_to_job(
            cfg.input_dir,
            cali_dataset_id=cali_id,
            test_dataset_id=test_id,
        )
        apply_test_dataset_nc(cfg)
        cfg.save()
        _init_manifest(cfg)
        report = validate_job_input(
            cfg.job_root,
            min_cali=cfg.min_cali_images,
            min_test=cfg.min_test_images,
            nc=cfg.nc,
            expect_imgsz=cfg.imgsz,
        )
        save_report(report, cfg.input_validation_path)
        write_input_manifest(cfg, validation_ok=report.ok)
        cfg.save()
        errors = [asdict(e) for e in report.errors]
        payload = _task_payload(cfg, validation_ok=report.ok, errors=errors or None)
        return (201 if report.ok else 400), payload
    except KeyError as e:
        _cleanup_job(cfg)
        raise _http_400(str(e)) from e
    except HTTPException:
        _cleanup_job(cfg)
        raise
    except OSError as e:
        _cleanup_job(cfg)
        raise _http_400(str(e)) from e
