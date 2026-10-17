#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate FPGA L43 decode coefficients from a YOLOv8-Pose ONNX graph.

L43 is not a normal convolution exported by Ultralytics.  It is an FPGA-side
helper table for the decoded part of the YOLO head:

    box decode:
        left/top     = (anchor_lt - distance_lt) * stride_bbox
        right/bottom = (anchor_rb + distance_rb) * stride_bbox

    keypoint decode:
        keypoint_xy  = (2 * raw_xy + anchor_kpt) * stride_kpt

Older versions of this script looked up ONNX constants by literal names such as
``/model.30/Constant_15_output_0``.  Those names are exporter artifacts: they can
change when the class count, exporter, simplifier, or quantization graph changes.

This implementation instead finds constants by graph structure and tensor
shape, starting from the final decoded output.  It still assumes the YOLOv8-Pose
decode math above, but it does not assume specific Constant node numbers.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import onnx
from onnx import numpy_helper

TOUT = 32
DEFAULT_WIDTH = 34000


@dataclass(frozen=True)
class DecodeConstants:
    """Constants needed by FPGA L43.

    Shapes after normalization:
      - anchor_lt:   [2, width]
      - anchor_rb:   [2, width]
      - stride_bbox: [width]
      - anchor_kpt:  [2, width]
      - stride_kpt:  [width]
    """

    anchor_lt: np.ndarray
    anchor_rb: np.ndarray
    stride_bbox: np.ndarray
    anchor_kpt: np.ndarray
    stride_kpt: np.ndarray
    source_names: dict[str, str]


def _shape_of_value(model: onnx.ModelProto) -> dict[str, tuple[int | str, ...]]:
    """Return known shapes for graph values, graph inputs and graph outputs."""
    out: dict[str, tuple[int | str, ...]] = {}
    for value in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        dims = []
        for dim in value.type.tensor_type.shape.dim:
            dims.append(dim.dim_value if dim.dim_value > 0 else dim.dim_param)
        out[value.name] = tuple(dims)
    return out


def _initializer_map(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    """Return ONNX initializers as numpy arrays."""
    return {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}


def _producer_map(model: onnx.ModelProto) -> dict[str, onnx.NodeProto]:
    """Map every produced tensor name to its producer node."""
    out: dict[str, onnx.NodeProto] = {}
    for node in model.graph.node:
        for name in node.output:
            out[name] = node
    return out


def _is_const_name(name: str, init: dict[str, np.ndarray]) -> bool:
    return name in init


def _norm_shape(shape: Sequence[int | str] | None) -> tuple[int | str, ...] | None:
    return tuple(shape) if shape is not None else None


def _same_shape(shape: Sequence[int | str] | None, expected: Sequence[int]) -> bool:
    return _norm_shape(shape) == tuple(expected)


def _const_input(
    node: onnx.NodeProto,
    init: dict[str, np.ndarray],
    *,
    shape: Sequence[int] | None = None,
) -> tuple[str, np.ndarray] | None:
    """Return the first initializer input on a node, optionally constrained by shape."""
    for name in node.input:
        if not _is_const_name(name, init):
            continue
        arr = init[name]
        if shape is not None and tuple(arr.shape) != tuple(shape):
            continue
        return name, arr
    return None


def _nonconst_inputs(node: onnx.NodeProto, init: dict[str, np.ndarray]) -> list[str]:
    return [name for name in node.input if not _is_const_name(name, init)]


def _walk_upstream(
    tensor_name: str,
    producers: dict[str, onnx.NodeProto],
) -> Iterable[onnx.NodeProto]:
    """Depth-first upstream traversal from a tensor to its producer dependencies."""
    seen_tensors: set[str] = set()
    seen_nodes: set[str] = set()
    stack = [tensor_name]
    while stack:
        name = stack.pop()
        if name in seen_tensors:
            continue
        seen_tensors.add(name)
        node = producers.get(name)
        if node is None:
            continue
        node_key = node.name or "|".join(node.output)
        if node_key not in seen_nodes:
            seen_nodes.add(node_key)
            yield node
        stack.extend(node.input)


def _find_final_concat(
    model: onnx.ModelProto,
    shapes: dict[str, tuple[int | str, ...]],
    width: int | None,
) -> tuple[onnx.NodeProto, str, str, str, int, int]:
    """Find the final decoded output concat.

    Expected final YOLOv8-Pose decoded output layout is:

        [box4, class_nc, keypoint6] along channel axis

    The function returns:

        final_concat, box_tensor, class_tensor, kpt_tensor, inferred_nc, inferred_width
    """
    graph_outputs = {out.name for out in model.graph.output}
    candidates: list[onnx.NodeProto] = []
    for node in model.graph.node:
        if node.op_type != "Concat":
            continue
        if any(out in graph_outputs for out in node.output):
            candidates.append(node)
    if not candidates:
        raise RuntimeError("Cannot find final Concat producing a graph output")

    for node in candidates:
        if len(node.input) != 3:
            continue
        input_shapes = [shapes.get(name) for name in node.input]
        if any(s is None or len(s) != 3 for s in input_shapes):
            continue
        channel_dims = [int(s[1]) for s in input_shapes]  # type: ignore[arg-type]
        widths = [int(s[2]) for s in input_shapes]  # type: ignore[arg-type]
        if len(set(widths)) != 1:
            continue
        w = widths[0]
        if width is not None and w != width:
            continue
        if channel_dims[0] != 4 or channel_dims[2] != 6:
            continue
        return node, node.input[0], node.input[1], node.input[2], channel_dims[1], w

    detail = [
        (node.name, list(node.input), [shapes.get(name) for name in node.input])
        for node in candidates
    ]
    raise RuntimeError(f"Cannot identify final decoded YOLO output concat: {detail}")


def _find_bbox_constants(
    box_tensor: str,
    producers: dict[str, onnx.NodeProto],
    shapes: dict[str, tuple[int | str, ...]],
    init: dict[str, np.ndarray],
    width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    """Find bbox anchors and stride from the upstream graph of final box output."""
    source_names: dict[str, str] = {}

    # Final box output is expected to be a Mul:
    #     concat(center_xy, wh) * stride_bbox
    box_mul = producers.get(box_tensor)
    if box_mul is None or box_mul.op_type != "Mul":
        raise RuntimeError(f"Box tensor {box_tensor!r} is not produced by a Mul node")
    stride_pair = _const_input(box_mul, init, shape=(1, width))
    if stride_pair is None:
        raise RuntimeError("Cannot find bbox stride constant [1,width] on final box Mul")
    source_names["stride_bbox"] = stride_pair[0]
    stride_bbox = stride_pair[1].reshape(width).astype(np.float32)

    anchor_lt = None
    anchor_rb = None
    for node in _walk_upstream(box_tensor, producers):
        if node.op_type not in {"Sub", "Add"}:
            continue
        if not any(_same_shape(shapes.get(out), (1, 2, width)) for out in node.output):
            continue
        c = _const_input(node, init, shape=(1, 2, width))
        if c is None:
            continue
        name, arr = c
        # The left/top branch is anchor - distance, so the constant is the first
        # input of a Sub.  The right/bottom branch is anchor + distance, so it is
        # the constant input of an Add.
        if node.op_type == "Sub" and node.input and node.input[0] == name:
            anchor_lt = arr.reshape(2, width).astype(np.float32)
            source_names["anchor_lt"] = name
        elif node.op_type == "Add":
            anchor_rb = arr.reshape(2, width).astype(np.float32)
            source_names["anchor_rb"] = name

    if anchor_lt is None:
        raise RuntimeError("Cannot find bbox left/top anchor constant via Sub(anchor, distance)")
    if anchor_rb is None:
        raise RuntimeError("Cannot find bbox right/bottom anchor constant via Add(anchor, distance)")
    return anchor_lt, anchor_rb, stride_bbox, source_names


def _find_keypoint_constants(
    kpt_tensor: str,
    producers: dict[str, onnx.NodeProto],
    shapes: dict[str, tuple[int | str, ...]],
    init: dict[str, np.ndarray],
    width: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """Find keypoint anchor and stride from the upstream graph of final kpt output."""
    source_names: dict[str, str] = {}
    anchor_kpt = None
    stride_kpt = None

    for node in _walk_upstream(kpt_tensor, producers):
        if node.op_type == "Mul" and any(
            _same_shape(shapes.get(out), (1, 2, 2, width)) for out in node.output
        ):
            c = _const_input(node, init, shape=(1, width))
            if c is not None:
                source_names["stride_kpt"] = c[0]
                stride_kpt = c[1].reshape(width).astype(np.float32)
        elif node.op_type == "Add" and any(
            _same_shape(shapes.get(out), (1, 2, 2, width)) for out in node.output
        ):
            c = _const_input(node, init, shape=(2, width))
            if c is not None:
                source_names["anchor_kpt"] = c[0]
                anchor_kpt = c[1].reshape(2, width).astype(np.float32)

    if anchor_kpt is None:
        raise RuntimeError("Cannot find keypoint anchor constant [2,width] via Add")
    if stride_kpt is None:
        raise RuntimeError("Cannot find keypoint stride constant [1,width] via Mul")
    return anchor_kpt, stride_kpt, source_names


def extract_l43_coeffs_from_onnx(model_path: str | os.PathLike, width: int | None = DEFAULT_WIDTH) -> DecodeConstants:
    """Extract L43 constants without relying on exporter-generated Constant names."""
    model = onnx.load(str(model_path), load_external_data=False)
    shapes = _shape_of_value(model)
    init = _initializer_map(model)
    producers = _producer_map(model)

    final_concat, box_tensor, cls_tensor, kpt_tensor, nc, inferred_width = _find_final_concat(
        model, shapes, width
    )
    anchor_lt, anchor_rb, stride_bbox, bbox_names = _find_bbox_constants(
        box_tensor, producers, shapes, init, inferred_width
    )
    anchor_kpt, stride_kpt, kpt_names = _find_keypoint_constants(
        kpt_tensor, producers, shapes, init, inferred_width
    )

    source_names = {
        "final_concat": final_concat.name,
        "box_tensor": box_tensor,
        "class_tensor": cls_tensor,
        "keypoint_tensor": kpt_tensor,
        "inferred_nc": str(nc),
        "inferred_width": str(inferred_width),
        **bbox_names,
        **kpt_names,
    }
    return DecodeConstants(
        anchor_lt=anchor_lt,
        anchor_rb=anchor_rb,
        stride_bbox=stride_bbox,
        anchor_kpt=anchor_kpt,
        stride_kpt=stride_kpt,
        source_names=source_names,
    )


def build_l43_wt(constants: DecodeConstants) -> np.ndarray:
    """Build the physical L43 tensor.

    Output shape is [32, 1, width].  Only channels 0..15 are meaningful:

      0..1   bbox lt coefficients k1 = -stride
      2..3   bbox lt offsets      b1 = anchor_lt * stride
      4..5   bbox rb coefficients k2 = +stride
      6..7   bbox rb offsets      b2 = anchor_rb * stride
      8..11  kpt xy coefficients k3 = 2 * stride
      12..15 kpt xy offsets      b3 = anchor_kpt * stride
      16..31 padding for T32 alignment
    """
    width = constants.stride_bbox.size
    if constants.stride_kpt.size != width:
        raise ValueError("bbox stride width and keypoint stride width differ")

    new_wt = np.zeros((TOUT, 1, width), dtype=np.float32)

    k1 = np.stack([-constants.stride_bbox, -constants.stride_bbox], axis=0)
    b1 = constants.anchor_lt * constants.stride_bbox
    k2 = np.stack([constants.stride_bbox, constants.stride_bbox], axis=0)
    b2 = constants.anchor_rb * constants.stride_bbox
    k3 = np.stack([2.0 * constants.stride_kpt] * 4, axis=0)
    b3 = np.stack(
        [
            constants.anchor_kpt[0] * constants.stride_kpt,
            constants.anchor_kpt[1] * constants.stride_kpt,
            constants.anchor_kpt[0] * constants.stride_kpt,
            constants.anchor_kpt[1] * constants.stride_kpt,
        ],
        axis=0,
    )

    new_wt[0:2, 0, :] = k1
    new_wt[2:4, 0, :] = b1
    new_wt[4:6, 0, :] = k2
    new_wt[6:8, 0, :] = b2
    new_wt[8:12, 0, :] = k3
    new_wt[12:16, 0, :] = b3
    return new_wt


def parallel_of_feature(input_arr: np.ndarray, ch: int, h: int, w: int, tout: int = TOUT) -> np.ndarray:
    """Pack [C,H,W] feature-like data into FPGA T32 layout [C/T32,H,W,T32]."""
    slice_of_ch = (ch + tout - 1) // tout
    src = np.asarray(input_arr).reshape(ch, h, w)
    padded = np.zeros((slice_of_ch * tout, h, w), dtype=src.dtype)
    padded[:ch] = src
    return padded.reshape(slice_of_ch, tout, h, w).transpose(0, 2, 3, 1)


def fp16array_to_bin(input_array: np.ndarray, output_file: str | os.PathLike) -> None:
    fp16_array = np.asarray(input_array, dtype=np.float32).astype(np.float16)
    fp16_array.tofile(output_file)


def generate_l43_wt_bin(
    model_path: str | os.PathLike,
    output_bin: str | os.PathLike,
    output_txt: str | os.PathLike | None = None,
    width: int | None = DEFAULT_WIDTH,
) -> dict[str, object]:
    """Generate T32_T32_L43_WT.bin from an ONNX model."""
    constants = extract_l43_coeffs_from_onnx(model_path, width=width)
    new_wt = build_l43_wt(constants)
    physical_width = new_wt.shape[2]
    packed = parallel_of_feature(new_wt, TOUT, 1, physical_width)

    output_bin = str(output_bin)
    os.makedirs(os.path.dirname(os.path.abspath(output_bin)) or ".", exist_ok=True)
    fp16array_to_bin(packed.flatten(), output_bin)

    if output_txt:
        np.savetxt(str(output_txt), packed.flatten(), fmt="%f")

    return {
        "output_bin": output_bin,
        "output_txt": str(output_txt) if output_txt else None,
        "bytes": os.path.getsize(output_bin),
        "elems": int(packed.size),
        "width": int(physical_width),
        "source_names": constants.source_names,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="从 ONNX 图结构生成 FPGA L43 WT bin")
    parser.add_argument("onnx_path", help="输入 ONNX 模型路径")
    parser.add_argument("output_bin", nargs="?", default="T32_T32_L43_WT.bin")
    parser.add_argument("--txt", dest="output_txt", default=None, help="可选：同时输出 txt 调试文件")
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help="候选点数量；默认 34000。传 0 可完全按 ONNX 推断。",
    )
    args = parser.parse_args()

    if not os.path.exists(args.onnx_path):
        raise FileNotFoundError(f"ONNX 文件不存在: {args.onnx_path}")

    width = None if args.width == 0 else args.width
    info = generate_l43_wt_bin(args.onnx_path, args.output_bin, args.output_txt, width=width)
    print(f"ONNX: {args.onnx_path}")
    print(f"输出: {info['output_bin']} ({info['bytes']} 字节, {info['elems']} 元素)")
    print(f"候选点数量: {info['width']}")
    print("图结构定位结果:")
    for key, value in sorted(info["source_names"].items()):
        print(f"  {key}: {value}")
    if info["output_txt"]:
        print(f"TXT : {info['output_txt']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
