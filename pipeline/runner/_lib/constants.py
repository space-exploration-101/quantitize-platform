#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""轻量常量（避免 Web 启动时拉入 eval_common / onnxruntime）。"""

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# test_data 原图边长；pose 标签里 w/h 按该尺寸归一化（cx/cy/关键点仍相对当前图）
DEFAULT_POSE_LABEL_WH_REF = 2048
