#!/usr/bin/env python3
"""Regression tests for exact QDQ activation export in generate_bin."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ENGINE = Path(__file__).resolve().parents[2] / "engine"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from check_certain_layer_multi import (  # noqa: E402
    GenerateBinTensorContractError,
    instrument_conv_inputs,
)


def _qdq_conv_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 1, 1, 4])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 1, 4])
    scale = numpy_helper.from_array(np.array(0.25, dtype=np.float32), "scale")
    zero = numpy_helper.from_array(np.array(128, dtype=np.uint8), "zero")
    weight = numpy_helper.from_array(np.ones((1, 1, 1, 1), dtype=np.float32), "weight")
    nodes = [
        helper.make_node("QuantizeLinear", ["images", "scale", "zero"], ["images_q"], name="input_q"),
        helper.make_node("DequantizeLinear", ["images_q", "scale", "zero"], ["images_dq"], name="input_dq"),
        helper.make_node("Conv", ["images_dq", "weight"], ["output"], name="/model.0/conv/Conv"),
    ]
    graph = helper.make_graph(nodes, "qdq-conv", [x], [y], [scale, zero, weight])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    return model


class GenerateBinQDQTests(unittest.TestCase):
    def test_conv_input_is_exact_dequantized_tensor(self) -> None:
        model, added = instrument_conv_inputs(_qdq_conv_model())
        self.assertEqual(added, ["images_dq"])

        session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        source = np.array([[[[0.12, 0.37, 0.62, 0.88]]]], dtype=np.float32)
        values = dict(zip((item.name for item in session.get_outputs()), session.run(None, {"images": source})))

        expected = np.array([[[[0.0, 0.25, 0.5, 1.0]]]], dtype=np.float32)
        np.testing.assert_array_equal(values["images_dq"], expected)
        self.assertFalse(np.array_equal(values["images_dq"], source))

    def test_instrumentation_does_not_modify_source_model(self) -> None:
        source = _qdq_conv_model()
        instrumented, _ = instrument_conv_inputs(source)
        self.assertEqual([item.name for item in source.graph.output], ["output"])
        self.assertEqual([item.name for item in instrumented.graph.output], ["output", "images_dq"])

    def test_unresolved_conv_input_reports_node_and_tensor(self) -> None:
        model = _qdq_conv_model()
        model.graph.node[2].input[0] = "missing_activation"
        with self.assertRaisesRegex(
            GenerateBinTensorContractError,
            r"/model\.0/conv/Conv: missing_activation",
        ):
            instrument_conv_inputs(model)


if __name__ == "__main__":
    unittest.main()
