#!/usr/bin/env python3
"""Regression tests for the legacy weight-only QDQ contract."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ENGINE = Path(__file__).resolve().parents[2] / "engine"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

import _load_local_onnxruntime_modules  # noqa: E402,F401
from onnxruntime.quantization import (  # noqa: E402
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization import qdq_quantizer, quant_utils, registry  # noqa: E402
from onnxruntime.quantization.operators import conv  # noqa: E402

from quantitize import validate_weight_only_qdq  # noqa: E402


class _Reader(CalibrationDataReader):
    def __init__(self) -> None:
        self.done = False

    def get_next(self):
        if self.done:
            return None
        self.done = True
        return {"images": np.array([[[[0.25, 0.75]]]], dtype=np.float16)}


def _fp16_conv_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("images", TensorProto.FLOAT16, [1, 1, 1, 2])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1, 2, 1, 2])
    weights = numpy_helper.from_array(
        np.array([[[[0.0077]]], [[[0.125]]]], dtype=np.float16),
        "weight",
    )
    node = helper.make_node("Conv", ["images", "weight"], ["output"], name="conv")
    graph = helper.make_graph([node], "weight-only", [x], [y], [weights])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    return model


class WeightOnlyQuantizationTests(unittest.TestCase):
    def test_local_patch_rebinds_eager_ort_registry_and_scale_helper(self) -> None:
        self.assertIs(registry.QDQRegistry["Conv"], conv.QDQConv)
        self.assertIs(
            qdq_quantizer.compute_data_quant_params.__globals__["compute_scale_zp"],
            quant_utils.compute_scale_zp,
        )

    def test_fp16_conv_quantizes_weights_without_activation_qdq(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.onnx"
            output = Path(tmp) / "quantized.onnx"
            onnx.save(_fp16_conv_model(), source)
            quantize_static(
                model_input=source,
                model_output=output,
                calibration_data_reader=_Reader(),
                quant_format=QuantFormat.QDQ,
                op_types_to_quantize=["Conv"],
                activation_type=QuantType.QInt16,
                weight_type=QuantType.QInt8,
                per_channel=True,
                extra_options={
                    "OpTypesToExcludeOutputQuantization": ["Conv"],
                    "QuantizeBias": False,
                },
            )

            validate_weight_only_qdq(source, output)
            model = onnx.load(output)
            self.assertFalse(any(node.op_type == "QuantizeLinear" for node in model.graph.node))
            dq_nodes = [node for node in model.graph.node if node.op_type == "DequantizeLinear"]
            self.assertEqual(len(dq_nodes), 1)
            initializers = {
                item.name: numpy_helper.to_array(item) for item in model.graph.initializer
            }
            scales = initializers[dq_nodes[0].input[1]]
            quantized = initializers[dq_nodes[0].input[0]]
            self.assertFalse(np.any(scales == 1.0))
            self.assertTrue(np.all(np.any(quantized != 0, axis=(1, 2, 3))))


if __name__ == "__main__":
    unittest.main()
