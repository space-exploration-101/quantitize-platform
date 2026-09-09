"""
Quantitize 前端 BFF：只渲染页面和转发表单，业务一律走 JSON API。

环境变量:
  API_BASE_URL   默认 http://127.0.0.1:8000
  WEB_DEMO_MODE  仅当设为 1/true/yes 时用演示数据；正式模式为 0
"""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
DEMO_MODE = os.environ.get("WEB_DEMO_MODE", "0").strip().lower() in ("1", "true", "yes")
WEB_DIR = Path(__file__).resolve().parent

PREPROCESS_MODES = [
    ("rgb", "RGB（彩色预训练）"),
    ("grayscale_uniform", "黑白三通道同值"),
    ("grayscale_r_channel", "黑白 R 通道（G=B=0）"),
    ("passthrough", "不做通道预处理（保持原图通道）"),
]

app = FastAPI(title="Quantitize Web UI")
templates = Jinja2Templates(directory=str(WEB_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")


class ApiError(Exception):
    def __init__(self, message: str, status_code: int = 502, body: Any = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.body = body


def _demo_datasets():
    return [
        SimpleNamespace(id="demo_cali", display_name="演示标定集", image_count=400),
    ], [
        SimpleNamespace(id="demo_test", display_name="演示测试集", image_count=100),
    ]


def _as_dataset_objs(items: list) -> list:
    out = []
    for d in items or []:
        if isinstance(d, dict):
            out.append(
                SimpleNamespace(
                    id=d.get("id", ""),
                    display_name=d.get("display_name") or d.get("name") or d.get("id", ""),
                    image_count=d.get("image_count", 0),
                )
            )
        else:
            out.append(d)
    return out


def _catalog_rows(catalog: list) -> list:
    rows = []
    for item in catalog or []:
        if not isinstance(item, dict):
            rows.append(item)
            continue
        entry = item.get("entry") or {}
        rows.append(
            SimpleNamespace(
                kind_label=item.get("kind_label", ""),
                download_kind=item.get("download_kind", ""),
                entry=SimpleNamespace(
                    id=entry.get("id", ""),
                    display_name=entry.get("display_name") or entry.get("name", ""),
                    image_count=entry.get("image_count"),
                    rel_path=entry.get("rel_path") or entry.get("path", ""),
                    note=entry.get("note", ""),
                ),
            )
        )
    return rows


def _error_items(payload: Any) -> list:
    if isinstance(payload, dict):
        if payload.get("errors"):
            out = []
            for e in payload["errors"]:
                if isinstance(e, dict):
                    out.append(SimpleNamespace(message=e.get("message") or json.dumps(e, ensure_ascii=False)))
                else:
                    out.append(SimpleNamespace(message=str(e)))
            return out
        detail = payload.get("detail")
        if isinstance(detail, str):
            return [SimpleNamespace(message=detail)]
        if detail is not None:
            return [SimpleNamespace(message=json.dumps(detail, ensure_ascii=False))]
    return [SimpleNamespace(message=str(payload))]


def _ns(**kwargs) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)


async def api_request(
    method: str,
    path: str,
    *,
    timeout: float = 30.0,
    **kwargs,
) -> httpx.Response:
    url = f"{API_BASE_URL}{path}"
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            return await client.request(method, url, **kwargs)
    except httpx.RequestError as e:
        raise ApiError(f"无法连接后端 API（{API_BASE_URL}）: {e}", status_code=502) from e


async def api_get_json(path: str, *, timeout: float = 30.0) -> dict:
    r = await api_request("GET", path, timeout=timeout)
    if r.status_code >= 400:
        raise ApiError(_detail_text(r), status_code=r.status_code if r.status_code != 404 else 404, body=_try_json(r))
    try:
        return r.json()
    except json.JSONDecodeError as e:
        raise ApiError("后端返回了非 JSON 响应", status_code=502) from e


def _try_json(r: httpx.Response) -> Any:
    try:
        return r.json()
    except Exception:
        return r.text


def _detail_text(r: httpx.Response) -> str:
    data = _try_json(r)
    if isinstance(data, dict):
        if data.get("detail"):
            d = data["detail"]
            return d if isinstance(d, str) else json.dumps(d, ensure_ascii=False)
        if data.get("errors"):
            return "; ".join(
                (e.get("message") if isinstance(e, dict) else str(e)) for e in data["errors"]
            )
    text = r.text.strip()
    return text[:500] if text else f"HTTP {r.status_code}"


def _html_error(request: Request, message: str, status_code: int = 502) -> HTMLResponse:
    safe = html.escape(message or "")
    if status_code >= 500:
        title, heading = "后端错误", "后端不可用"
        hint = f"<p>请确认 API 已启动（API_BASE_URL={html.escape(API_BASE_URL)}）。</p>"
    elif status_code == 404:
        title, heading = "未找到", "任务不存在"
        hint = ""
    else:
        title, heading = "请求无效", "无法打开任务"
        hint = ""
    return HTMLResponse(
        f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>{title}</title><link rel="stylesheet" href="/static/style.css"></head>
<body><nav><a href="/">新建任务</a><a href="/datasets">共享数据集</a><a href="/history">历史任务</a></nav>
<main><h1>{heading}</h1><p class="errors">{safe}</p>
{hint}
<p><a href="/">返回首页</a></p></main></body></html>""",
        status_code=status_code,
    )


@app.exception_handler(ApiError)
async def _handle_api_error(request: Request, exc: ApiError):
    return _html_error(request, exc.message, exc.status_code)


@app.get("/health")
async def health():
    if DEMO_MODE:
        return {"web": "ok", "demo": True, "api": None, "api_base": API_BASE_URL}
    try:
        data = await api_get_json("/health")
        return {"web": "ok", "demo": False, "api": data, "api_base": API_BASE_URL}
    except ApiError as e:
        return {"web": "ok", "demo": False, "api": {"ok": False, "error": e.message}, "api_base": API_BASE_URL}


async def _worker_and_datasets():
    worker = await api_get_json("/api/worker")
    datasets = await api_get_json("/api/datasets")
    return worker, datasets


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    errors = None
    busy = False
    current_job = None
    if DEMO_MODE:
        cali, test = _demo_datasets()
    else:
        try:
            worker, datasets = await _worker_and_datasets()
            busy = bool(worker.get("busy"))
            current_job = worker.get("current_job")
            cali = _as_dataset_objs(datasets.get("cali", []))
            test = _as_dataset_objs(datasets.get("test", []))
        except ApiError as e:
            return _html_error(request, e.message, e.status_code)
    return templates.TemplateResponse(
        request,
        "new_task.html",
        {
            "busy": busy,
            "current_job": current_job,
            "cali_datasets": cali,
            "test_datasets": test,
            "preprocess_modes": PREPROCESS_MODES,
            "preprocess_mode": "grayscale_uniform",
            "errors": errors,
            "display_name": "",
        },
    )


@app.get("/datasets", response_class=HTMLResponse)
async def datasets_page(request: Request):
    if DEMO_MODE:
        rows = _catalog_rows(
            [
                {
                    "kind_label": "标定",
                    "download_kind": "cali",
                    "entry": {"id": "demo_cali", "display_name": "演示标定集", "image_count": 400, "path": "(demo)"},
                },
                {
                    "kind_label": "测试",
                    "download_kind": "test",
                    "entry": {"id": "demo_test", "display_name": "演示测试集", "image_count": 100, "path": "(demo)"},
                },
            ]
        )
        return templates.TemplateResponse(
            request,
            "datasets.html",
            {"rows": rows, "shared_root": "(demo)", "busy": False, "current_job": None},
        )
    try:
        worker = await api_get_json("/api/worker")
        data = await api_get_json("/api/datasets")
    except ApiError as e:
        return _html_error(request, e.message, e.status_code)
    return templates.TemplateResponse(
        request,
        "datasets.html",
        {
            "rows": _catalog_rows(data.get("catalog") or []),
            "shared_root": f"via API {API_BASE_URL}",
            "busy": bool(worker.get("busy")),
            "current_job": worker.get("current_job"),
        },
    )


@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request):
    if DEMO_MODE:
        return templates.TemplateResponse(
            request, "history.html", {"rows": [], "busy": False, "current_job": None}
        )
    try:
        worker = await api_get_json("/api/worker")
        data = await api_get_json("/api/tasks")
    except ApiError as e:
        return _html_error(request, e.message, e.status_code)
    return templates.TemplateResponse(
        request,
        "history.html",
        {
            "rows": data.get("tasks", []),
            "busy": bool(worker.get("busy")),
            "current_job": worker.get("current_job"),
        },
    )


@app.get("/tasks/{task_id}", response_class=HTMLResponse)
async def monitor(request: Request, task_id: str):
    if DEMO_MODE:
        return HTMLResponse("<p>演示模式：监控页需连接后端 API。</p><p><a href='/'>返回</a></p>")
    task = await api_get_json(f"/api/tasks/{task_id}")
    cfg = _ns(
        job_id=task.get("task_id", task_id),
        display_name=task.get("display_name", task_id),
        onnx_name=task.get("onnx_name", ""),
        preprocess_mode=task.get("preprocess_mode", ""),
        cali_dataset_id=task.get("cali_dataset_id", ""),
        test_dataset_id=task.get("test_dataset_id", ""),
    )
    return templates.TemplateResponse(
        request,
        "monitor.html",
        {
            "cfg": cfg,
            "manifest": task.get("manifest") or {},
            "steps": task.get("steps") or [],
            "log_tail": "",
            "busy": task.get("busy"),
            "current_job": task.get("current_job"),
            "worker_error": task.get("worker_error"),
            "task_error": task.get("task_error"),
            "worker_status": task.get("worker_status"),
            "gpu_readiness": task.get("gpu_readiness") or {},
            "status": task.get("status"),
            "has_zip": task.get("has_zip"),
            "preprocess_mode_label": task.get("preprocess_mode"),
        },
    )


async def _forward_create(path: str, data: dict, files: list) -> httpx.Response:
    return await api_request("POST", path, timeout=600.0, data=data, files=files)


@app.post("/tasks")
async def create_zip(
    request: Request,
    display_name: str = Form(""),
    onnx_name: str = Form(""),
    preprocess_mode: str = Form("grayscale_uniform"),
    zip_file: UploadFile = File(...),
):
    if DEMO_MODE:
        return _html_error(request, "演示模式不能创建任务，请设置 WEB_DEMO_MODE=0", 400)
    content = await zip_file.read()
    files = {"zip_file": (zip_file.filename or "upload.zip", content, zip_file.content_type or "application/zip")}
    data = {
        "display_name": display_name,
        "onnx_name": onnx_name,
        "preprocess_mode": preprocess_mode,
    }
    r = await _forward_create("/api/tasks", data, files)
    payload = _try_json(r)
    if r.status_code >= 400:
        cali, test = _demo_datasets()
        try:
            worker, datasets = await _worker_and_datasets()
            cali = _as_dataset_objs(datasets.get("cali", []))
            test = _as_dataset_objs(datasets.get("test", []))
            busy = bool(worker.get("busy"))
            current_job = worker.get("current_job")
        except ApiError:
            busy, current_job = False, None
        return templates.TemplateResponse(
            request,
            "new_task.html",
            {
                "busy": busy,
                "current_job": current_job,
                "cali_datasets": cali,
                "test_datasets": test,
                "preprocess_modes": PREPROCESS_MODES,
                "preprocess_mode": preprocess_mode,
                "errors": _error_items(payload),
                "display_name": display_name,
            },
            status_code=400,
        )
    task_id = payload.get("task_id") if isinstance(payload, dict) else None
    if not task_id:
        return _html_error(request, "创建成功但未返回 task_id", 502)
    return RedirectResponse(url=f"/tasks/{task_id}", status_code=303)


@app.post("/tasks/shared")
async def create_shared(
    request: Request,
    display_name: str = Form(""),
    onnx_name: str = Form(""),
    preprocess_mode: str = Form("grayscale_uniform"),
    cali_dataset_id: str = Form(...),
    test_dataset_id: str = Form(...),
    model_pt: UploadFile = File(...),
):
    if DEMO_MODE:
        return _html_error(request, "演示模式不能创建任务，请设置 WEB_DEMO_MODE=0", 400)
    content = await model_pt.read()
    files = {"model_pt": (model_pt.filename or "model.pt", content, "application/octet-stream")}
    data = {
        "display_name": display_name,
        "onnx_name": onnx_name,
        "preprocess_mode": preprocess_mode,
        "cali_dataset_id": cali_dataset_id,
        "test_dataset_id": test_dataset_id,
        "imgsz": "1280",
    }
    r = await _forward_create("/api/tasks/shared", data, files)
    payload = _try_json(r)
    if r.status_code >= 400:
        try:
            worker, datasets = await _worker_and_datasets()
            cali = _as_dataset_objs(datasets.get("cali", []))
            test = _as_dataset_objs(datasets.get("test", []))
            busy = bool(worker.get("busy"))
            current_job = worker.get("current_job")
        except ApiError:
            cali, test = [], []
            busy, current_job = False, None
        return templates.TemplateResponse(
            request,
            "new_task.html",
            {
                "busy": busy,
                "current_job": current_job,
                "cali_datasets": cali,
                "test_datasets": test,
                "preprocess_modes": PREPROCESS_MODES,
                "preprocess_mode": preprocess_mode,
                "errors": _error_items(payload),
                "display_name": display_name,
            },
            status_code=400,
        )
    task_id = payload.get("task_id") if isinstance(payload, dict) else None
    if not task_id:
        return _html_error(request, "创建成功但未返回 task_id", 502)
    return RedirectResponse(url=f"/tasks/{task_id}", status_code=303)


@app.post("/tasks/{task_id}/start")
async def start(request: Request, task_id: str):
    if DEMO_MODE:
        return _html_error(request, "演示模式不能启动任务", 400)
    r = await api_request("POST", f"/api/tasks/{task_id}/start", timeout=60.0)
    if r.status_code >= 400:
        return _html_error(request, _detail_text(r), r.status_code)
    return RedirectResponse(url=f"/tasks/{task_id}", status_code=303)


@app.post("/tasks/{task_id}/cancel-wait")
async def cancel_wait(request: Request, task_id: str):
    if DEMO_MODE:
        return _html_error(request, "演示模式不能操作任务", 400)
    r = await api_request("POST", f"/api/tasks/{task_id}/cancel-wait", timeout=30.0)
    if r.status_code >= 400:
        return _html_error(request, _detail_text(r), r.status_code)
    return RedirectResponse(url=f"/tasks/{task_id}", status_code=303)


@app.get("/tasks/{task_id}/metrics", response_class=HTMLResponse)
async def metrics_page(request: Request, task_id: str):
    data = await api_get_json(f"/api/tasks/{task_id}/metrics")
    empty_groups = {"class_error": [], "fp": [], "fn": []}
    cfg = _ns(
        job_id=data.get("task_id", task_id),
        display_name=data.get("display_name", task_id),
    )
    return templates.TemplateResponse(
        request,
        "metrics.html",
        {
            "cfg": cfg,
            "pt": data.get("pt"),
            "onnx": data.get("onnx"),
            "fpga": data.get("fpga"),
            "pt_success": data.get("pt_success") or [],
            "pt_errors": data.get("pt_errors") or empty_groups,
            "onnx_errors": data.get("onnx_errors") or empty_groups,
            "fpga_errors": data.get("fpga_errors") or empty_groups,
            "preprocess_mode_label": data.get("preprocess_mode") or "",
            "eval_data": data.get("eval_data") or {},
            "busy": False,
            "current_job": None,
        },
    )


@app.get("/tasks/{task_id}/bundle_inventory", response_class=HTMLResponse)
async def bundle_inventory_page(request: Request, task_id: str):
    task = await api_get_json(f"/api/tasks/{task_id}")
    inv = await api_get_json(f"/api/tasks/{task_id}/bundle_inventory")
    cfg = _ns(
        job_id=task.get("task_id", task_id),
        display_name=task.get("display_name", task_id),
        test_dataset_id=task.get("test_dataset_id") or "",
        cali_dataset_id=task.get("cali_dataset_id") or "",
        onnx_name=task.get("onnx_name") or "",
    )
    onnx = cfg.onnx_name or "model"
    slim_hint = (
        "以下为成果包体积（已排除 {onnx}.onnx、*_fp16*.onnx、workspace/bin/ 等非交付文件）。"
        "主推理 ONNX 为 {onnx}_output.onnx。"
    ).format(onnx=onnx)
    return templates.TemplateResponse(
        request,
        "bundle_inventory.html",
        {
            "cfg": cfg,
            "inv": inv,
            "source_label": inv.get("source_label", ""),
            "slim_hint": slim_hint,
            "has_zip": inv.get("has_zip") or task.get("has_zip"),
            "busy": False,
            "current_job": None,
        },
    )


@app.get("/tasks/{task_id}/download")
async def download_zip(task_id: str):
    r = await api_request("GET", f"/api/tasks/{task_id}/download", timeout=120.0)
    if r.status_code >= 400:
        raise ApiError(_detail_text(r), status_code=r.status_code)
    headers = {}
    if "content-disposition" in r.headers:
        headers["content-disposition"] = r.headers["content-disposition"]
    return Response(
        content=r.content,
        media_type=r.headers.get("content-type", "application/zip"),
        headers=headers,
        status_code=r.status_code,
    )


@app.get("/tasks/{task_id}/input_manifest")
@app.get("/tasks/{task_id}/input_manifest.md")
async def input_manifest(task_id: str):
    r = await api_request("GET", f"/api/tasks/{task_id}/input_manifest", timeout=30.0)
    if r.status_code >= 400:
        raise ApiError(_detail_text(r), status_code=r.status_code)
    return Response(content=r.content, media_type=r.headers.get("content-type", "text/markdown"))


@app.get("/tasks/{task_id}/results/{eval_name}/{kind}/{filename}")
async def proxy_overlay(task_id: str, eval_name: str, kind: str, filename: str):
    r = await api_request(
        "GET",
        f"/api/tasks/{task_id}/results/{eval_name}/{kind}/{filename}",
        timeout=30.0,
    )
    if r.status_code >= 400:
        raise ApiError(_detail_text(r), status_code=r.status_code)
    return Response(
        content=r.content,
        media_type=r.headers.get("content-type", "application/octet-stream"),
        headers={"Cache-Control": "no-cache, max-age=0, must-revalidate"},
    )
