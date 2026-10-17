#!/usr/bin/env python3
"""Export one image's FPGA-layer activations from a Pose21 Gray1 ONNX model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import onnx

# Keep the production runtime isolation used by the other engine scripts.
sys.path = [p for p in sys.path if p and "isaac-sim" not in p.lower()]
import _pin_stdlib_platform  # noqa: E402,F401
import _setup_local_onnxruntime  # noqa: E402,F401
import onnxruntime as ort  # noqa: E402

from grayscale_preprocess import (  # noqa: E402
    PREPROCESS_MODE_ALIASES,
    normalize_preprocess_mode,
    preprocess_by_mode,
)


TILE_CHANNELS = 32
SCHEMA_VERSION = 1
HEAD_BRANCHES = (
    *(f"/model.30/cv2.{i}/cv2.{i}" for i in range(4)),
    *(f"/model.30/cv3.{i}/cv3.{i}" for i in range(4)),
    *(f"/model.30/cv4.{i}/cv4.{i}" for i in range(4)),
)
HEAD_AUXILIARY_NODES = (
    "/model.30/dfl/Transpose",
    "/model.30/dfl/Softmax",
    "/model.30/dfl/Reshape",
    "/model.30/dfl/Reshape_1",
    "/model.30/Mul_2",
    "/model.30/Sigmoid",
    "/model.30/Mul_3",
    "/model.30/Concat",
    "/model.30/Reshape_9",
    "/model.30/Concat_5",
)


class LayerTraceError(RuntimeError):
    """Raised when a deterministic layer trace cannot be produced."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _node_map(model: onnx.ModelProto) -> dict[str, onnx.NodeProto]:
    return {node.name: node for node in model.graph.node if node.name}


def primary_graph_output(model: onnx.ModelProto) -> str:
    names = [item.name for item in model.graph.output]
    if not names:
        raise LayerTraceError("model has no graph output")
    return "output0" if "output0" in names else names[0]


def _is_runtime_tensor(
    name: str,
    initializers: set[str],
    producers: dict[str, onnx.NodeProto],
) -> bool:
    if not name or name in initializers:
        return False
    producer = producers.get(name)
    return producer is None or producer.op_type != "Constant"


def _operation(
    node: onnx.NodeProto,
    initializers: set[str],
    producers: dict[str, onnx.NodeProto],
) -> dict[str, Any]:
    # Conv input[1:] are parameters, including weight-only DQ outputs. They are
    # already delivered by the quantization bundle and are not image activations.
    candidate_inputs = node.input[:1] if node.op_type == "Conv" else node.input
    return {
        "node_name": node.name,
        "op_type": node.op_type,
        "inputs": [
            name
            for name in candidate_inputs
            if _is_runtime_tensor(name, initializers, producers)
        ],
        "outputs": list(node.output),
    }


def _new_layer(trace_id: str, module: int, kind: str) -> dict[str, Any]:
    return {
        "trace_id": trace_id,
        "yolo_module": module,
        "kind": kind,
        "operations": [],
    }


def build_hardware_layers(model: onnx.ModelProto) -> list[dict[str, Any]]:
    """Build the stable T32 compute-layer map plus explicit routing records."""
    nodes = list(model.graph.node)
    by_name = _node_map(model)
    initializers = {item.name for item in model.graph.initializer}
    producers = {output: node for node in nodes for output in node.output}
    layers: list[dict[str, Any]] = []

    for module in range(30):
        prefix = f"/model.{module}/"
        sub_index = 0
        route_index = 0
        current_compute: dict[str, Any] | None = None
        for node in nodes:
            if not node.name.startswith(prefix):
                continue
            if node.op_type in {"QuantizeLinear", "DequantizeLinear", "Constant"}:
                continue
            if node.op_type == "Conv" and node.name.endswith("/conv/Conv"):
                current_compute = _new_layer(
                    f"T32_L{module:02d}_s{sub_index:02d}", module, "compute"
                )
                current_compute["operations"].append(
                    _operation(node, initializers, producers)
                )
                activation = by_name.get(node.name[: -len("conv/Conv")] + "act/Mul")
                if activation is not None:
                    current_compute["operations"].append(
                        _operation(activation, initializers, producers)
                    )
                layers.append(current_compute)
                sub_index += 1
                continue
            if node.op_type == "Add" and current_compute is not None:
                current_compute["operations"].append(
                    _operation(node, initializers, producers)
                )
                continue
            if node.op_type in {"MaxPool", "Resize"}:
                current_compute = _new_layer(
                    f"T32_L{module:02d}_s{sub_index:02d}", module, node.op_type.lower()
                )
                current_compute["operations"].append(
                    _operation(node, initializers, producers)
                )
                layers.append(current_compute)
                sub_index += 1
                continue
            if node.op_type == "Concat":
                route = _new_layer(
                    f"T32_L{module:02d}_r{route_index:02d}", module, "routing"
                )
                route["operations"].append(_operation(node, initializers, producers))
                layers.append(route)
                route_index += 1

    for branch_index, base in enumerate(HEAD_BRANCHES):
        layer_id = 31 + branch_index
        for sub_index in range(3):
            suffix = f".{sub_index}"
            conv_name = f"{base}{suffix}/conv/Conv" if sub_index < 2 else f"{base}{suffix}/Conv"
            conv = by_name.get(conv_name)
            if conv is None:
                raise LayerTraceError(f"required Pose head node is missing: {conv_name}")
            layer = _new_layer(f"T32_L{layer_id:02d}_s{sub_index:02d}", 30, "head")
            layer["operations"].append(_operation(conv, initializers, producers))
            if sub_index < 2:
                activation = by_name.get(f"{base}{suffix}/act/Mul")
                if activation is None:
                    raise LayerTraceError(
                        f"required Pose head activation is missing: {base}{suffix}/act/Mul"
                    )
                layer["operations"].append(
                    _operation(activation, initializers, producers)
                )
            layers.append(layer)

    for route_index, name in enumerate(HEAD_AUXILIARY_NODES):
        node = by_name.get(name)
        if node is None:
            continue
        route = _new_layer(f"T32_L30_r{route_index:02d}", 30, "head_auxiliary")
        route["operations"].append(_operation(node, initializers, producers))
        layers.append(route)

    selected_nodes = {
        operation["node_name"]
        for layer in layers
        for operation in layer["operations"]
    }
    final_route_index = len(HEAD_AUXILIARY_NODES)
    for output_name in (primary_graph_output(model),):
        producer = producers.get(output_name)
        if producer is None or producer.name in selected_nodes:
            continue
        route = _new_layer(
            f"T32_L30_r{final_route_index:02d}", 30, "model_output"
        )
        route["operations"].append(_operation(producer, initializers, producers))
        layers.append(route)
        selected_nodes.add(producer.name)
        final_route_index += 1

    if not layers:
        raise LayerTraceError("no FPGA layer boundaries were found in the model")
    return layers


def required_tensor_names(layers: Iterable[dict[str, Any]], graph_input: str) -> list[str]:
    names: OrderedDict[str, None] = OrderedDict([(graph_input, None)])
    for layer in layers:
        for operation in layer["operations"]:
            for name in operation["inputs"] + operation["outputs"]:
                names.setdefault(name, None)
    return list(names)


def expose_tensors(
    model: onnx.ModelProto, tensor_names: Iterable[str]
) -> tuple[onnx.ModelProto, list[str]]:
    instrumented = onnx.ModelProto()
    instrumented.CopyFrom(model)
    try:
        instrumented = onnx.shape_inference.infer_shapes(instrumented)
    except Exception as exc:
        raise LayerTraceError(f"ONNX shape inference failed: {exc}") from exc

    info_by_name = {
        info.name: info
        for info in (
            list(instrumented.graph.input)
            + list(instrumented.graph.value_info)
            + list(instrumented.graph.output)
        )
    }
    existing = {info.name for info in instrumented.graph.output}
    graph_inputs = {info.name for info in instrumented.graph.input}
    unresolved = []
    exposed = []
    for name in tensor_names:
        if name in graph_inputs:
            continue
        info = info_by_name.get(name)
        if info is None or not info.type.HasField("tensor_type"):
            unresolved.append(name)
            continue
        if name not in existing:
            clone = onnx.ValueInfoProto()
            clone.CopyFrom(info)
            instrumented.graph.output.append(clone)
            existing.add(name)
        exposed.append(name)
    if unresolved:
        sample = ", ".join(unresolved[:10])
        raise LayerTraceError(
            f"cannot expose {len(unresolved)} tensors after shape inference: {sample}"
        )
    return instrumented, exposed


def pack_t32(array: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Convert a batch-1 NCHW/NC tensor to the FPGA channel-last T32 layout."""
    data = np.asarray(array).astype("<f2", copy=False)
    logical_shape = list(data.shape)
    metadata: dict[str, Any] = {
        "logical_shape": logical_shape,
        "storage_dtype": "float16",
        "byte_order": "little",
        "tile_channels": TILE_CHANNELS,
    }
    if data.ndim not in (3, 4) or data.shape[0] != 1:
        packed = np.ascontiguousarray(data)
        metadata.update(
            {
                "layout": "RAW_CONTIGUOUS",
                "physical_shape": list(packed.shape),
                "valid_channels": None,
                "padded_channels": None,
            }
        )
        return packed, metadata

    channels = int(data.shape[1])
    channel_tiles = math.ceil(channels / TILE_CHANNELS)
    padded_channels = channel_tiles * TILE_CHANNELS
    if data.ndim == 4:
        _, _, height, width = data.shape
        padded = np.zeros((padded_channels, height, width), dtype="<f2")
        padded[:channels] = data[0]
        packed = padded.reshape(channel_tiles, TILE_CHANNELS, height, width).transpose(
            0, 2, 3, 1
        )
        layout = "T32_CHW32"
    else:
        _, _, candidates = data.shape
        padded = np.zeros((padded_channels, candidates), dtype="<f2")
        padded[:channels] = data[0]
        packed = padded.reshape(channel_tiles, TILE_CHANNELS, candidates).transpose(2, 0, 1)
        if channel_tiles == 1:
            packed = packed[:, 0, :]
        layout = "T32_NC32"
    packed = np.ascontiguousarray(packed)
    metadata.update(
        {
            "layout": layout,
            "physical_shape": list(packed.shape),
            "valid_channels": channels,
            "padded_channels": padded_channels - channels,
        }
    )
    return packed, metadata


def unpack_t32(packed: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    """Inverse of pack_t32, used by tests and export-time verification."""
    logical_shape = tuple(metadata["logical_shape"])
    layout = metadata["layout"]
    physical = np.asarray(packed).reshape(metadata["physical_shape"])
    if layout == "RAW_CONTIGUOUS":
        return physical.reshape(logical_shape)
    channels = int(metadata["valid_channels"])
    if layout == "T32_CHW32":
        _, _, height, width = logical_shape
        logical = physical.transpose(0, 3, 1, 2).reshape(-1, height, width)[:channels]
        return logical[np.newaxis]
    if layout == "T32_NC32":
        _, _, candidates = logical_shape
        if physical.ndim == 2:
            physical = physical[:, np.newaxis, :]
        logical = physical.transpose(1, 2, 0).reshape(-1, candidates)[:channels]
        return logical[np.newaxis]
    raise LayerTraceError(f"unsupported trace layout: {layout}")


def _stats(data: np.ndarray) -> dict[str, Any]:
    values = np.asarray(data, dtype=np.float32)
    finite = values[np.isfinite(values)]
    return {
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "mean": float(finite.mean()) if finite.size else None,
        "nan_count": int(np.isnan(values).sum()),
        "inf_count": int(np.isinf(values).sum()),
    }


def _safe_tensor_filename(index: int, owner: str, role: str, tensor_name: str) -> str:
    digest = hashlib.sha256(tensor_name.encode("utf-8")).hexdigest()[:10]
    return f"tensor_{index:04d}_{owner}_{role}_{digest}.bin"


def _tensor_owners(layers: list[dict[str, Any]], graph_input: str) -> dict[str, tuple[str, str]]:
    owners = {graph_input: ("network", "input")}
    for layer in layers:
        for operation in layer["operations"]:
            for index, name in enumerate(operation["inputs"]):
                owners.setdefault(name, (layer["trace_id"], f"input{index}"))
            for index, name in enumerate(operation["outputs"]):
                owners.setdefault(name, (layer["trace_id"], f"output{index}"))
    return owners


def _write_tensor(
    root: Path,
    index: int,
    tensor_name: str,
    array: np.ndarray,
    owner: tuple[str, str],
) -> dict[str, Any]:
    packed, packing = pack_t32(array)
    original_fp16 = np.asarray(array).astype("<f2", copy=False)
    restored = unpack_t32(packed, packing)
    if not np.array_equal(restored, original_fp16, equal_nan=True):
        raise LayerTraceError(f"T32 roundtrip mismatch for tensor: {tensor_name}")
    if packing["padded_channels"]:
        channels = int(packing["valid_channels"])
        if packing["layout"] == "T32_CHW32":
            padded_logical = packed.transpose(0, 3, 1, 2).reshape(
                -1, packed.shape[1], packed.shape[2]
            )
        else:
            shaped = packed[:, np.newaxis, :] if packed.ndim == 2 else packed
            padded_logical = shaped.transpose(1, 2, 0).reshape(-1, shaped.shape[0])
        if np.any(padded_logical[channels:] != 0):
            raise LayerTraceError(f"non-zero T32 padding for tensor: {tensor_name}")

    if owner == ("network", "input"):
        relative = Path("source/network_input.bin")
    else:
        relative = Path("tensors") / _safe_tensor_filename(index, owner[0], owner[1], tensor_name)
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    packed.tofile(path)
    expected_bytes = packed.size * np.dtype("<f2").itemsize
    if path.stat().st_size != expected_bytes:
        raise LayerTraceError(f"size verification failed for {relative}")
    return {
        "id": f"tensor_{index:04d}",
        "onnx_name": tensor_name,
        "file": relative.as_posix(),
        "source_dtype": str(np.asarray(array).dtype),
        **packing,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "statistics": _stats(original_fp16),
    }


def _replace_tensor_names(layers: list[dict[str, Any]], ids: dict[str, str]) -> None:
    for layer in layers:
        for operation in layer["operations"]:
            operation["input_tensor_ids"] = [ids[name] for name in operation.pop("inputs")]
            operation["output_tensor_ids"] = [ids[name] for name in operation.pop("outputs")]


def _write_layers_csv(path: Path, layers: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "trace_id",
                "yolo_module",
                "kind",
                "onnx_nodes",
                "op_types",
                "input_tensor_ids",
                "output_tensor_ids",
            ),
        )
        writer.writeheader()
        for layer in layers:
            operations = layer["operations"]
            writer.writerow(
                {
                    "trace_id": layer["trace_id"],
                    "yolo_module": layer["yolo_module"],
                    "kind": layer["kind"],
                    "onnx_nodes": ";".join(op["node_name"] for op in operations),
                    "op_types": ";".join(op["op_type"] for op in operations),
                    "input_tensor_ids": ";".join(
                        tensor_id
                        for op in operations
                        for tensor_id in op["input_tensor_ids"]
                    ),
                    "output_tensor_ids": ";".join(
                        tensor_id
                        for op in operations
                        for tensor_id in op["output_tensor_ids"]
                    ),
                }
            )


README_TEXT = """# FPGA Layer Trace

This directory contains one image's intermediate tensors from the Pose21 Gray1
weight-only ONNX model. `manifest.json` is the authoritative index.

Activation BIN files are little-endian float16. NCHW tensors use T32 channel
packing: `[1,C,H,W] -> [ceil(C/32),H,W,32]`. The decoded model output uses
`[1,C,N] -> [N,ceil(C/32),32]`; the single-tile `[1,31,34000]` output is stored
as `[34000,32]`. Padding lanes are zero and must not enter post-processing.

Compute layers retain the established `T32_Lxx_sxx` IDs. Routing-only records
use `T32_Lxx_rxx` so Concat and head decode tensors do not renumber weight/BN
layers. All inputs of multi-input operators are listed separately.
"""


def export_layer_trace(
    model_path: Path,
    image_path: Path,
    output_dir: Path,
    preprocess_mode: str = "passthrough",
) -> Path:
    model_path = model_path.resolve()
    image_path = image_path.resolve()
    output_dir = output_dir.resolve()
    if not model_path.is_file():
        raise LayerTraceError(f"model does not exist: {model_path}")
    if not image_path.is_file():
        raise LayerTraceError(f"image does not exist: {image_path}")
    if output_dir.exists():
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            raise LayerTraceError(f"refusing to overwrite existing path: {output_dir}")
    staging = output_dir.with_name(f"{output_dir.name}.partial.{os.getpid()}")
    if staging.exists():
        raise LayerTraceError(f"staging directory already exists: {staging}")
    staging.mkdir(parents=True)

    mode = normalize_preprocess_mode(preprocess_mode)
    try:
        model = onnx.load(str(model_path), load_external_data=True)
        onnx.checker.check_model(model)
        layers = build_hardware_layers(model)
        graph_inputs = [item.name for item in model.graph.input]
        graph_output_count = len(model.graph.output)
        primary_output_name = primary_graph_output(model)
        if len(graph_inputs) != 1:
            raise LayerTraceError(f"expected one graph input, got: {graph_inputs}")
        graph_input = graph_inputs[0]
        needed = required_tensor_names(layers, graph_input)
        instrumented, exposed = expose_tensors(model, needed)

        input_data = preprocess_by_mode(
            image_path, mode, target_size=(1280, 1280), dtype=np.float16
        )
        if input_data.shape != (1, 1, 1280, 1280):
            raise LayerTraceError(
                f"preprocess produced {input_data.shape}, expected (1,1,1280,1280)"
            )
        session = ort.InferenceSession(
            instrumented.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        if session.get_inputs()[0].name != graph_input:
            raise LayerTraceError("instrumented model input changed unexpectedly")
        output_names = [item.name for item in session.get_outputs()]
        results = session.run(exposed, {graph_input: input_data})
        values = {graph_input: input_data}
        values.update(dict(zip(exposed, results)))
        missing = [name for name in needed if name not in values]
        if missing:
            raise LayerTraceError(f"runtime did not return required tensors: {missing[:10]}")
        primary_output = values.get(primary_output_name)
        if primary_output is None or tuple(primary_output.shape) != (1, 31, 34000):
            shape = None if primary_output is None else tuple(primary_output.shape)
            raise LayerTraceError(
                f"Pose21 output contract failed: {primary_output_name} has shape {shape}, "
                "expected (1,31,34000)"
            )

        source_dir = staging / "source"
        source_dir.mkdir()
        source_image = source_dir / f"source_image{image_path.suffix.lower()}"
        shutil.copy2(image_path, source_image)

        owners = _tensor_owners(layers, graph_input)
        tensors = []
        ids = {}
        for index, name in enumerate(needed):
            item = _write_tensor(staging, index, name, values[name], owners[name])
            tensors.append(item)
            ids[name] = item["id"]
        _replace_tensor_names(layers, ids)

        payload = {
            "schema": "pose21-gray1-layer-trace/v1",
            "schema_version": SCHEMA_VERSION,
            "tool": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
                "arguments": {
                    "model": str(model_path),
                    "image": str(image_path),
                    "output_dir": str(output_dir),
                    "preprocess_mode": mode,
                },
            },
            "model": {
                "path": str(model_path),
                "sha256": sha256_file(model_path),
                "graph_input": graph_input,
                "primary_graph_output": primary_output_name,
                "source_graph_output_count": graph_output_count,
                "instrumented_output_count": len(output_names),
            },
            "image": {
                "path": str(image_path),
                "archived_file": source_image.relative_to(staging).as_posix(),
                "sha256": sha256_file(image_path),
            },
            "preprocess": {
                "mode": mode,
                "shape": [1, 1, 1280, 1280],
                "dtype": "float16",
                "normalization": "pixel / 255.0",
            },
            "runtime": {
                "provider": "CPUExecutionProvider",
                "onnxruntime": ort.__version__,
                "python": platform.python_version(),
            },
            "layer_count": len(layers),
            "tensor_count": len(tensors),
            "primary_output_tensor_id": ids[primary_output_name],
            "layers": layers,
            "tensors": tensors,
        }
        (staging / "manifest.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        _write_layers_csv(staging / "layers.csv", layers)
        (staging / "README.md").write_text(README_TEXT, encoding="ascii")

        checksum_lines = []
        for path in sorted(p for p in staging.rglob("*") if p.is_file()):
            if path.name == "SHA256SUMS":
                continue
            checksum_lines.append(
                f"{sha256_file(path)}  {path.relative_to(staging).as_posix()}"
            )
        (staging / "SHA256SUMS").write_text(
            "\n".join(checksum_lines) + "\n", encoding="ascii"
        )

        if output_dir.exists():
            output_dir.rmdir()
        os.replace(staging, output_dir)
        return output_dir
    except Exception:
        # Preserve the .partial directory as failure evidence.
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export one Gray1 image's FPGA-layer inputs and outputs"
    )
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--preprocess-mode",
        default="passthrough",
        choices=tuple(PREPROCESS_MODE_ALIASES),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        output = export_layer_trace(
            args.model, args.image, args.output_dir, args.preprocess_mode
        )
    except Exception as exc:
        print(f"layer trace export failed: {exc}", file=sys.stderr)
        return 1
    print(f"layer trace exported: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
