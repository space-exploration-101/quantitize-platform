#!/usr/bin/env python3
"""API contract tests for GPU liveness/readiness reporting."""

from __future__ import annotations

import sys
import unittest
import json
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.api.app import health, ready  # noqa: E402
from apps.api.gpu_readiness import GPUReadiness  # noqa: E402


def unavailable_gpu() -> GPUReadiness:
    return GPUReadiness(
        ready=False,
        reason="GPU utilization 100% exceeds the 90% threshold",
        required=True,
        min_free_mib=24576,
        max_utilization_percent=90,
        index=0,
        uuid="GPU-test",
        name="NVIDIA H200 NVL",
        memory_total_mib=143771,
        memory_free_mib=80000,
        utilization_percent=100,
    )


class APIGPURouteTests(unittest.TestCase):
    def test_liveness_does_not_fail_when_gpu_is_busy(self):
        response = health()
        self.assertTrue(response["ok"])

    @patch("apps.api.app.probe_gpu", side_effect=unavailable_gpu)
    def test_readiness_returns_503_when_gpu_is_busy(self, _probe):
        response = ready()
        self.assertEqual(response.status_code, 503)
        payload = json.loads(response.body)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["gpu"]["utilization_percent"], 100)


if __name__ == "__main__":
    unittest.main()
