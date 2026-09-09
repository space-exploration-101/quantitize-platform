"""
兼容旧引擎脚本的入口：``import _setup_local_onnxruntime``。

实际逻辑委托给 ``_load_local_onnxruntime_modules``（补丁目录为
``quantitize-platform/patches``）。
"""

from __future__ import annotations

import _load_local_onnxruntime_modules  # noqa: F401
