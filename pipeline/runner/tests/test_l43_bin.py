from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ENGINE = Path(__file__).resolve().parents[2] / "engine"
sys.path.insert(0, str(ENGINE))

from L43_bin_process import extract_l43_coeffs_from_onnx, generate_l43_wt_bin


def pose_decode_model(class_count=21):
    width = 3
    anchors = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.float32)
    stride = np.array([[8, 16, 32]], dtype=np.float32)
    initializers = [
        numpy_helper.from_array(anchors, "renamed_bbox_anchor"),
        numpy_helper.from_array(anchors[0] - 0.5, "renamed_kpt_anchor"),
        numpy_helper.from_array(stride, "renamed_stride"),
        numpy_helper.from_array(np.array([1, 4, width], dtype=np.int64), "reshape_shape"),
    ]
    nodes = [
        helper.make_node("Sub", ["renamed_bbox_anchor", "distance"], ["left_top"]),
        helper.make_node("Add", ["renamed_bbox_anchor", "distance"], ["right_bottom"]),
        helper.make_node("Concat", ["left_top", "right_bottom"], ["box_unscaled"], axis=1),
        helper.make_node("Mul", ["box_unscaled", "renamed_stride"], ["boxes"]),
        helper.make_node("Add", ["keypoints", "renamed_kpt_anchor"], ["keypoints_offset"]),
        helper.make_node("Mul", ["keypoints_offset", "renamed_stride"], ["keypoints_scaled"]),
        helper.make_node("Reshape", ["keypoints_scaled", "reshape_shape"], ["keypoints_flat"]),
        helper.make_node("Concat", ["keypoints_flat", "visibility"], ["keypoints_final"], axis=1),
        helper.make_node("Concat", ["boxes", "classes", "keypoints_final"], ["output"], axis=1),
    ]
    inputs = [
        helper.make_tensor_value_info("distance", TensorProto.FLOAT, [1, 2, width]),
        helper.make_tensor_value_info("keypoints", TensorProto.FLOAT, [1, 2, 2, width]),
        helper.make_tensor_value_info("visibility", TensorProto.FLOAT, [1, 2, width]),
        helper.make_tensor_value_info("classes", TensorProto.FLOAT, [1, class_count, width]),
    ]
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, class_count + 10, width])
    model = helper.make_model(helper.make_graph(nodes, "renamed-head", inputs, [output], initializers), opset_imports=[helper.make_opsetid("", 13)])
    return onnx.shape_inference.infer_shapes(model)


class L43BinTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / "model.onnx"
        onnx.save(pose_decode_model(), self.model)

    def test_extracts_constants_after_exporter_names_change(self):
        constants = extract_l43_coeffs_from_onnx(self.model, width=3)
        np.testing.assert_array_equal(constants.stride_bbox, [8, 16, 32])
        np.testing.assert_array_equal(constants.anchor_lt, [[1, 2, 3], [4, 5, 6]])
        self.assertEqual(constants.source_names["inferred_nc"], "21")

    def test_binary_coefficients_padding_and_size(self):
        output = self.root / "T32_T32_L43_WT.bin"
        result = generate_l43_wt_bin(self.model, output, width=3)
        packed = np.fromfile(output, dtype="<f2").reshape(3, 32)
        np.testing.assert_array_equal(packed[:, 0], [-8, -16, -32])
        np.testing.assert_array_equal(packed[:, 2], [8, 32, 96])
        np.testing.assert_array_equal(packed[:, 4], [8, 16, 32])
        np.testing.assert_array_equal(packed[:, 8], [16, 32, 64])
        np.testing.assert_array_equal(packed[:, 12], [4, 24, 80])
        self.assertFalse(np.any(packed[:, 16:]))
        self.assertEqual(result["bytes"], 3 * 32 * 2)

    def test_four_and_six_classes_are_not_confused_with_decode_channels(self):
        for class_count in [4, 6]:
            with self.subTest(class_count=class_count):
                onnx.save(pose_decode_model(class_count), self.model)
                constants = extract_l43_coeffs_from_onnx(self.model, width=3)
                self.assertEqual(constants.source_names["inferred_nc"], str(class_count))
                self.assertEqual(constants.source_names["keypoint_tensor"], "keypoints_final")

    def test_rejects_unexpected_width(self):
        with self.assertRaisesRegex(RuntimeError, "Cannot identify final decoded YOLO output concat"):
            extract_l43_coeffs_from_onnx(self.model, width=34000)


if __name__ == "__main__":
    unittest.main()
