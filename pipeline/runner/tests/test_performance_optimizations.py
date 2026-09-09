#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Byte-exact and lifecycle regression tests for H200 optimizations."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np

PIPELINE = Path(__file__).resolve().parents[2]
ENGINE = PIPELINE / "engine"
LIB = PIPELINE / "runner" / "_lib"
for path in (str(ENGINE), str(LIB), str(PIPELINE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import creat_bin  # noqa: E402
import job_config  # noqa: E402
import png_bin_converter  # noqa: E402
import scratch_lifecycle  # noqa: E402
from job_config import JobConfig  # noqa: E402


def _reference_pack(values: np.ndarray) -> bytes:
    values = np.asarray(values, dtype=np.uint16).reshape(-1)
    if values.size % 2:
        values = np.append(values, 0)
    out = bytearray()
    for left, right in values.reshape(-1, 2):
        v1 = int(left) & 0x0FFF
        v2 = int(right) & 0x0FFF
        out.extend((v1 & 0xFF, ((v2 & 0x0F) << 4) | ((v1 >> 8) & 0x0F), (v2 >> 4) & 0xFF))
    return bytes(out)


def _reference_feature(data: np.ndarray, channels: int, height: int, width: int) -> np.ndarray:
    out = np.zeros(((channels + 31) // 32, height, width, 32))
    for h in range(height):
        for w in range(width):
            for block in range(out.shape[0]):
                for lane in range(32):
                    channel = block * 32 + lane
                    out[block, h, w, lane] = data[channel, h, w] if channel < channels else 0
    return out


def _reference_weight(data: np.ndarray) -> np.ndarray:
    cout, cin, ky, kx = data.shape
    out = np.zeros(((cout + 31) // 32, (cin + 31) // 32, ky, kx, 32, 32))
    for ob in range(out.shape[0]):
        for ib in range(out.shape[1]):
            for y in range(ky):
                for x in range(kx):
                    for ot in range(32):
                        for it in range(32):
                            oc = ob * 32 + ot
                            ic = ib * 32 + it
                            if oc < cout and ic < cin:
                                out[ob, ib, y, x, ot, it] = data[oc, ic, y, x]
    return out


class VectorizedPrimitiveTests(unittest.TestCase):
    def test_12bit_pack_is_byte_exact(self) -> None:
        rng = np.random.default_rng(42)
        for size in (1, 2, 3, 31, 32, 33, 2000):
            values = rng.integers(0, 5000, size=size, dtype=np.uint16)
            packed = png_bin_converter.pack_12bit_to_bytes(values)
            self.assertEqual(packed, _reference_pack(values))
            restored = png_bin_converter.unpack_12bit_from_bytes(packed)[:size]
            np.testing.assert_array_equal(restored, values & 0x0FFF)

    def test_feature_reorder_matches_reference(self) -> None:
        for channels in (1, 31, 32, 33, 65):
            data = np.arange(channels * 3 * 4, dtype=np.float16).reshape(channels, 3, 4)
            expected = _reference_feature(data, channels, 3, 4)
            actual = creat_bin.parallel_of_feature(data, channels, 3, 4)
            np.testing.assert_array_equal(actual, expected)

    def test_weight_reorder_matches_reference(self) -> None:
        for cout, cin in ((3, 4), (33, 2), (2, 35)):
            data = np.arange(cout * cin * 2, dtype=np.int8).reshape(cout, cin, 1, 2)
            np.testing.assert_array_equal(creat_bin.parallel_weight(data, cout, cin, 1, 2), _reference_weight(data))

    def test_int8_pair_writer_is_byte_exact(self) -> None:
        values = np.array([-128, -1, 0, 1, 127, 42, -42], dtype=np.int16)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "weights.bin"
            creat_bin.INTarray_to_bin(values, output)
            padded = np.pad(values, (0, 1))
            self.assertEqual(output.read_bytes(), bytes(int(v) & 0xFF for v in padded))


class ScratchLifecycleTests(unittest.TestCase):
    def test_failed_scratch_is_verified_before_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            persistent = base / "persistent" / "task_1"
            scratch_root = base / "scratch"
            persistent.mkdir(parents=True)
            scratch_root.mkdir()
            cfg = JobConfig(job_root=persistent, job_id="task_1", onnx_name="model", use_scratch=True)
            with mock.patch.object(job_config, "TASK_SCRATCH_ROOT", scratch_root), mock.patch.object(
                scratch_lifecycle, "TASK_SCRATCH_ROOT", scratch_root
            ), mock.patch.dict(os.environ, {"TASK_SCRATCH_MIN_FREE_BYTES": "0"}):
                scratch_lifecycle.ensure_scratch_ready(cfg)
                payload = cfg.workspace_dir / "partial.bin"
                payload.write_bytes(b"failed-but-important")
                result = scratch_lifecycle.archive_failed_scratch(cfg)
                self.assertEqual(result["status"], "verified")
                self.assertFalse(cfg.scratch_dir.exists())
                archived = persistent / "failure_artifacts" / "workspace" / "partial.bin"
                self.assertEqual(archived.read_bytes(), b"failed-but-important")
                inventory = json.loads((persistent / "failure_inventory.json").read_text(encoding="utf-8"))
                self.assertEqual(inventory["status"], "verified")

    def test_success_cleanup_requires_valid_persistent_zip(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            persistent = base / "persistent" / "task_2"
            scratch_root = base / "scratch"
            persistent.mkdir(parents=True)
            scratch_root.mkdir()
            cfg = JobConfig(job_root=persistent, job_id="task_2", onnx_name="model", use_scratch=True)
            with mock.patch.object(job_config, "TASK_SCRATCH_ROOT", scratch_root), mock.patch.object(
                scratch_lifecycle, "TASK_SCRATCH_ROOT", scratch_root
            ), mock.patch.dict(os.environ, {"TASK_SCRATCH_MIN_FREE_BYTES": "0"}):
                scratch_lifecycle.ensure_scratch_ready(cfg)
                (cfg.workspace_dir / "done.bin").write_bytes(b"done")
                with self.assertRaises(RuntimeError):
                    scratch_lifecycle.finalize_success_scratch(cfg)
                self.assertTrue(cfg.scratch_dir.exists())
                with zipfile.ZipFile(cfg.bundle_zip_path(), "w") as zf:
                    zf.writestr("manifest.json", "{}")
                result = scratch_lifecycle.finalize_success_scratch(cfg)
                self.assertEqual(result["status"], "cleaned")
                self.assertFalse(cfg.scratch_dir.exists())


if __name__ == "__main__":
    unittest.main()
