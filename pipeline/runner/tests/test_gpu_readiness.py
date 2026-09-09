#!/usr/bin/env python3
"""Unit tests for the API GPU readiness gate."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.api.gpu_readiness import probe_gpu  # noqa: E402


class GPUReadinessTests(unittest.TestCase):
    def _completed(self, stdout: str = "", *, returncode: int = 0, stderr: str = ""):
        return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr=stderr)

    @patch.dict(os.environ, {}, clear=True)
    @patch("apps.api.gpu_readiness.subprocess.run")
    def test_ready_gpu(self, run):
        run.return_value = self._completed(
            "0, GPU-test, NVIDIA H200 NVL, 143771, 80000, 25\n"
        )
        result = probe_gpu()
        self.assertTrue(result.ready)
        self.assertEqual(result.uuid, "GPU-test")
        self.assertEqual(result.memory_free_mib, 80000)

    @patch.dict(os.environ, {"GPU_MIN_FREE_MIB": "30000"}, clear=True)
    @patch("apps.api.gpu_readiness.subprocess.run")
    def test_waits_for_free_memory(self, run):
        run.return_value = self._completed(
            "0, GPU-test, NVIDIA H200 NVL, 143771, 12000, 10\n"
        )
        result = probe_gpu()
        self.assertFalse(result.ready)
        self.assertIn("below", result.reason)

    @patch.dict(os.environ, {"GPU_MAX_UTILIZATION_PERCENT": "80"}, clear=True)
    @patch("apps.api.gpu_readiness.subprocess.run")
    def test_waits_for_utilization(self, run):
        run.return_value = self._completed(
            "0, GPU-test, NVIDIA H200 NVL, 143771, 80000, 95\n"
        )
        result = probe_gpu()
        self.assertFalse(result.ready)
        self.assertIn("exceeds", result.reason)

    @patch.dict(os.environ, {"GPU_READINESS_REQUIRED": "0"}, clear=True)
    @patch("apps.api.gpu_readiness.subprocess.run")
    def test_gate_can_be_disabled_explicitly(self, run):
        result = probe_gpu()
        self.assertTrue(result.ready)
        run.assert_not_called()

    @patch.dict(os.environ, {}, clear=True)
    @patch("apps.api.gpu_readiness.subprocess.run")
    def test_query_failure_is_not_ready(self, run):
        run.return_value = self._completed(returncode=1, stderr="driver unavailable")
        result = probe_gpu()
        self.assertFalse(result.ready)
        self.assertEqual(result.error, "driver unavailable")


if __name__ == "__main__":
    unittest.main()
