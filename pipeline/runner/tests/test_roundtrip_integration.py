from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline" / "runner" / "_lib"))

from apps.api import artifacts
from apps.web import app as web_app
from fastapi.testclient import TestClient
from job_config import JobConfig
from manifest import load_manifest
import runner_core


class RoundtripIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cfg = JobConfig(self.root, "test", "model", preprocess_mode="passthrough")

    def summary(self, stage, count):
        directory = self.cfg.results_dir / stage
        directory.mkdir(parents=True)
        summary = {"n_images": count, "precision": 1.0, "recall": 1.0, "f1": 1.0, "mAP": None}
        (directory / "summary.json").write_text(json.dumps(summary))
        errors = {"groups": {"fp": [{"image": "case.png", "overlay": "case.png", "FP": 1, "TP": 0}]}}
        (directory / "error_cases.json").write_text(json.dumps(errors))
        (directory / "error_cases").mkdir()
        (directory / "error_cases" / "case.png").write_bytes(b"sample")

    def test_metrics_prefer_new_directory_and_serve_its_overlay(self):
        self.summary("fpga_eval", 1)
        self.summary("fpga_input_roundtrip_eval", 2)
        result = artifacts.metrics_payload(self.cfg)
        self.assertEqual(result["fpga"]["n_images"], 2)
        self.assertEqual(result["fpga_eval_name"], "fpga_input_roundtrip_eval")
        response = artifacts.overlay_file_response(self.cfg, result["fpga_eval_name"], "error_cases", "case.png")
        self.assertEqual(Path(response.path).parent.parent.name, "fpga_input_roundtrip_eval")
        self.assertTrue(result["fpga_errors"]["fp"][0]["overlay_v"])

    def test_metrics_preserve_legacy_directory(self):
        self.summary("fpga_eval", 1)
        result = artifacts.metrics_payload(self.cfg)
        self.assertEqual(result["fpga"]["n_images"], 1)
        self.assertEqual(result["fpga_eval_name"], "fpga_eval")
        self.assertEqual(len(result["fpga_errors"]["fp"]), 1)

    def test_invalid_new_summary_falls_back_to_legacy(self):
        self.summary("fpga_eval", 1)
        directory = self.cfg.results_dir / "fpga_input_roundtrip_eval"
        directory.mkdir()
        (directory / "summary.json").write_text("invalid JSON")
        self.assertEqual(artifacts.metrics_payload(self.cfg)["fpga_eval_name"], "fpga_eval")

    def test_web_uses_stage_from_api_for_overlay_url(self):
        self.summary("fpga_input_roundtrip_eval", 2)
        payload = artifacts.metrics_payload(self.cfg)
        with patch.object(web_app, "api_get_json", new=AsyncMock(return_value=payload)):
            response = TestClient(web_app.app).get("/tasks/test/metrics")
        self.assertEqual(response.status_code, 200)
        self.assertIn("/results/fpga_input_roundtrip_eval/error_cases/case.png", response.text)

    def test_web_falls_back_for_older_api_payloads(self):
        self.summary("fpga_eval", 1)
        payload = artifacts.metrics_payload(self.cfg)
        payload.pop("fpga_eval_name")
        with patch.object(web_app, "api_get_json", new=AsyncMock(return_value=payload)):
            response = TestClient(web_app.app).get("/tasks/test/metrics")
        self.assertIn("/results/fpga_eval/error_cases/case.png", response.text)

    def test_legacy_completed_stage_resumes_without_repeating_evaluation(self):
        path = self.root / "manifest.json"
        path.write_text(json.dumps({"steps": {"fpga_eval": {"status": "completed"}}}))
        manifest = load_manifest(path)
        self.assertTrue(runner_core._step_completed(manifest, "fpga_input_roundtrip_eval"))
        self.assertNotIn("fpga_eval", manifest["steps"])
        self.assertEqual(json.loads(path.read_text())["steps"]["fpga_eval"]["status"], "completed")

    def test_existing_canonical_stage_is_not_overwritten(self):
        path = self.root / "manifest.json"
        path.write_text(json.dumps({"steps": {"fpga_eval": {"status": "completed"}, "fpga_input_roundtrip_eval": {"status": "failed"}}}))
        self.assertFalse(runner_core._step_completed(load_manifest(path), "fpga_input_roundtrip_eval"))

    def test_runner_passes_preprocessing_mode_to_bin_export(self):
        with patch.object(runner_core, "run_script", return_value=SimpleNamespace(returncode=0)) as execute:
            runner_core.step_generate_bin(self.cfg, {})
        self.assertEqual(execute.call_args.args[1], [str(self.cfg.output_onnx()), str(self.cfg.layer_bin_dir()), "--preprocess-mode", "passthrough"])

    def test_merge_passes_model_to_l43_generation(self):
        output = self.cfg.all_bin_dir()

        def execute(script, arguments, **kwargs):
            if script == runner_core.engine_script(runner_core.EngineScripts.MERGE_WT):
                (output / "ALL_wt.bin").write_bytes(b"merged")
            return SimpleNamespace(returncode=0)

        with patch.object(runner_core, "run_script", side_effect=execute) as mocked:
            runner_core.step_merge_bin(self.cfg, {})
        self.assertEqual(mocked.call_args_list[0].args[1][-1], str(self.cfg.output_onnx()))
        self.assertEqual((output / "ALL_WT.bin").read_bytes(), b"merged")

    def test_roundtrip_updates_canonical_stage(self):
        manifest = {}
        with patch.object(runner_core, "run_script", return_value=SimpleNamespace(returncode=0)):
            runner_core.step_onnx_eval(self.cfg, manifest, input_mode="fpga_side_view")
        self.assertEqual(manifest["steps"]["fpga_input_roundtrip_eval"]["status"], "completed")
        self.assertNotIn("fpga_eval", manifest["steps"])


if __name__ == "__main__":
    unittest.main()
