#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Web 页面 smoke（需已启动 uvicorn）。"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request


def fetch(url: str) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return resp.status, resp.read(500).decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read(200).decode("utf-8", errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser(description="Web smoke test")
    parser.add_argument("--base", default="http://127.0.0.1:8765")
    parser.add_argument("--task-id", default="_e2e_smoke")
    args = parser.parse_args()
    base = args.base.rstrip("/")
    checks = [
        (f"{base}/", 200, "新建任务"),
        (f"{base}/history", 200, "历史"),
        (f"{base}/tasks/{args.task_id}", 200, "监控"),
        (f"{base}/tasks/{args.task_id}/metrics", 200, "精度"),
    ]
    ok = True
    for url, expect, name in checks:
        code, _ = fetch(url)
        mark = "PASS" if code == expect else "FAIL"
        print(f"[{mark}] {name}: HTTP {code} (expect {expect})")
        if code != expect:
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
