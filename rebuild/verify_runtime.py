#!/usr/bin/env python3
"""Validate the rebuilt Quantitize API Python/GPU runtime."""

import json
import platform
import sys


def main() -> int:
    failures = []
    report = {
        "python": platform.python_version(),
        "machine": platform.machine(),
    }

    if platform.machine() not in {"x86_64", "AMD64"}:
        failures.append("current rebuild assets require amd64/x86_64")

    try:
        import torch

        report["torch"] = torch.__version__
        report["torch_cuda"] = torch.version.cuda
        report["cuda_available"] = torch.cuda.is_available()
        report["torch_arch_list"] = (
            torch.cuda.get_arch_list() if torch.cuda.is_available() else []
        )
        report["gpu_names"] = (
            [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available()
            else []
        )
        if not torch.cuda.is_available():
            failures.append("torch CUDA is unavailable")
        if "sm_90" not in report["torch_arch_list"]:
            failures.append("torch binary does not contain sm_90 required by NVIDIA H200")
    except Exception as exc:
        failures.append(f"torch import/runtime failed: {exc}")

    try:
        import onnxruntime as ort

        report["onnxruntime"] = ort.__version__
        report["onnxruntime_providers"] = ort.get_available_providers()
        if "CUDAExecutionProvider" not in report["onnxruntime_providers"]:
            failures.append("ONNX Runtime CUDAExecutionProvider is unavailable")
    except Exception as exc:
        failures.append(f"onnxruntime import/runtime failed: {exc}")

    expected_import_versions = {
        "numpy": "2.0.1",
        "cv2": "4.13.0",
        "ultralytics": "8.4.6",
        "fastapi": "0.128.8",
        "uvicorn": "0.39.0",
    }
    for module, expected in expected_import_versions.items():
        try:
            imported = __import__(module)
            actual = getattr(imported, "__version__", "imported")
            report[f"module_{module}"] = actual
            if actual != expected:
                failures.append(
                    f"{module} import version is {actual}, expected {expected}"
                )
        except Exception as exc:
            failures.append(f"{module} import failed: {exc}")

    report["failures"] = failures
    report["ok"] = not failures
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
