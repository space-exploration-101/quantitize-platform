#!/usr/bin/env python3
"""Unit tests for the deterministic FPGA layer-trace exporter."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

ENGINE = Path(__file__).resolve().parents[2] / "engine"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from export_layer_trace import (  # noqa: E402
    build_hardware_layers,
    pack_t32,
    required_tensor_names,
    unpack_t32,
)


class T32PackingTests(unittest.TestCase):
    def _roundtrip(self, shape: tuple[int, ...]) -> tuple[np.ndarray, dict]:
        source = np.arange(np.prod(shape), dtype=np.float16).reshape(shape)
        packed, metadata = pack_t32(source)
        restored = unpack_t32(packed, metadata)
        np.testing.assert_array_equal(restored, source)
        return packed, metadata

    def test_nchw_padding_and_roundtrip(self) -> None:
        packed, metadata = self._roundtrip((1, 33, 2, 3))
        self.assertEqual(metadata["layout"], "T32_CHW32")
        self.assertEqual(metadata["physical_shape"], [2, 2, 3, 32])
        self.assertEqual(metadata["padded_channels"], 31)
        logical = packed.transpose(0, 3, 1, 2).reshape(64, 2, 3)
        self.assertFalse(np.any(logical[33:]))

    def test_final_output_layout(self) -> None:
        packed, metadata = self._roundtrip((1, 31, 7))
        self.assertEqual(metadata["layout"], "T32_NC32")
        self.assertEqual(metadata["physical_shape"], [7, 32])
        self.assertFalse(np.any(packed[:, 31]))

    def test_raw_layout_for_non_feature_tensor(self) -> None:
        packed, metadata = self._roundtrip((4, 5))
        self.assertEqual(metadata["layout"], "RAW_CONTIGUOUS")
        self.assertEqual(packed.shape, (4, 5))


class LayerMappingTests(unittest.TestCase):
    def _model(self) -> onnx.ModelProto:
        nodes = [
            helper.make_node("Conv", ["images", "w0"], ["c0"], name="/model.0/conv/Conv"),
            helper.make_node("Sigmoid", ["c0"], ["sig0"], name="/model.0/act/Sigmoid"),
            helper.make_node("Mul", ["c0", "sig0"], ["a0"], name="/model.0/act/Mul"),
            helper.make_node("Concat", ["a0", "skip"], ["cat"], name="/model.1/Concat", axis=1),
            helper.make_node("Conv", ["cat", "w1"], ["c1"], name="/model.2/cv1/conv/Conv"),
            helper.make_node("Sigmoid", ["c1"], ["sig1"], name="/model.2/cv1/act/Sigmoid"),
            helper.make_node("Mul", ["c1", "sig1"], ["a1"], name="/model.2/cv1/act/Mul"),
            helper.make_node("Add", ["a1", "skip"], ["add"], name="/model.2/m.0/Add"),
        ]
        # Add the required 12 x 3 Pose head nodes.
        previous = "add"
        head_weights = []
        for branch in ("cv2", "cv3", "cv4"):
            for scale in range(4):
                base = f"/model.30/{branch}.{scale}/{branch}.{scale}"
                for sub in range(3):
                    weight = f"w_{branch}_{scale}_{sub}"
                    head_weights.append(helper.make_tensor(weight, TensorProto.FLOAT, [1], [1.0]))
                    if sub < 2:
                        conv_out = f"{branch}_{scale}_{sub}_conv"
                        nodes.append(helper.make_node("Conv", [previous, weight], [conv_out], name=f"{base}.{sub}/conv/Conv"))
                        sig = f"{branch}_{scale}_{sub}_sig"
                        out = f"{branch}_{scale}_{sub}_out"
                        nodes.append(helper.make_node("Sigmoid", [conv_out], [sig], name=f"{base}.{sub}/act/Sigmoid"))
                        nodes.append(helper.make_node("Mul", [conv_out, sig], [out], name=f"{base}.{sub}/act/Mul"))
                        previous = out
                    else:
                        out = f"{branch}_{scale}_{sub}_out"
                        nodes.append(helper.make_node("Conv", [previous, weight], [out], name=f"{base}.{sub}/Conv"))
                        previous = out
        nodes.append(
            helper.make_node(
                "Identity", [previous], ["output0"], name="/model.30/final/Identity"
            )
        )
        graph = helper.make_graph(
            nodes,
            "trace-test",
            [
                helper.make_tensor_value_info("images", TensorProto.FLOAT16, [1, 1, 2, 2]),
                helper.make_tensor_value_info("skip", TensorProto.FLOAT16, [1, 1, 2, 2]),
            ],
            [helper.make_tensor_value_info("output0", TensorProto.FLOAT16, None)],
            initializer=[
                helper.make_tensor("w0", TensorProto.FLOAT, [1], [1.0]),
                helper.make_tensor("w1", TensorProto.FLOAT, [1], [1.0]),
                *head_weights,
            ],
        )
        return helper.make_model(graph)

    def test_compute_ids_and_multi_input_routing(self) -> None:
        model = self._model()
        layers = build_hardware_layers(model)
        by_id = {layer["trace_id"]: layer for layer in layers}
        self.assertIn("T32_L00_s00", by_id)
        self.assertIn("T32_L01_r00", by_id)
        self.assertIn("T32_L02_s00", by_id)
        self.assertIn("T32_L30_r10", by_id)
        self.assertEqual(
            by_id["T32_L01_r00"]["operations"][0]["inputs"], ["a0", "skip"]
        )
        add_names = [
            operation["node_name"]
            for operation in by_id["T32_L02_s00"]["operations"]
        ]
        self.assertIn("/model.2/m.0/Add", add_names)
        conv_operation = by_id["T32_L00_s00"]["operations"][0]
        self.assertEqual(conv_operation["inputs"], ["images"])
        needed = required_tensor_names(layers, "images")
        self.assertEqual(needed[0], "images")
        self.assertIn("output0", needed)
        self.assertEqual(len(needed), len(set(needed)))


if __name__ == "__main__":
    unittest.main()
