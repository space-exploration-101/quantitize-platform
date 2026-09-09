#!/usr/bin/env python3
"""State-transition tests for the single-job GPU waiting worker."""

from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.api.gpu_readiness import GPUReadiness  # noqa: E402
from apps.api import worker  # noqa: E402


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


class GPUWaitingWorkerTests(unittest.TestCase):
    @patch("apps.api.worker.probe_gpu", side_effect=unavailable_gpu)
    def test_waiting_job_can_be_cancelled_back_to_pending(self, _probe):
        with tempfile.TemporaryDirectory() as temp:
            job_dir = Path(temp) / "job-1"
            job_dir.mkdir()
            (job_dir / "manifest.json").write_text(
                json.dumps({"steps": {"pt_eval": {"status": "pending"}}}),
                encoding="utf-8",
            )

            result = worker.start_pipeline(job_dir)
            self.assertIsNotNone(result)
            self.assertEqual(result["status"], "waiting_gpu")
            self.assertTrue(worker.cancel_waiting("job-1"))

            deadline = time.monotonic() + 2
            while worker.is_busy() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertFalse(worker.is_busy())
            manifest = json.loads((job_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "pending")


if __name__ == "__main__":
    unittest.main()
