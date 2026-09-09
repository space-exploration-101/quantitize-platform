"""
强制加载 patches/ 下的 onnxruntime 量化补丁（quant_utils.py、operators/conv.py）。

在导入 onnxruntime.quantization 之前使用::

    import _load_local_onnxruntime_modules  # noqa
    from onnxruntime.quantization import quantize_static
"""

from __future__ import annotations

import importlib.util
import os
import sys
import warnings
from pathlib import Path


def _platform_root() -> Path:
    # pipeline/engine/_load_... -> quantitize-platform
    return Path(__file__).resolve().parents[2]


def load_local_onnxruntime_modules(patches_dir: str | Path | None = None) -> None:
    root = _platform_root()
    patches = Path(patches_dir) if patches_dir else (root / "patches")
    quant_utils_path = patches / "onnxruntime" / "quantization" / "quant_utils.py"
    conv_path = patches / "onnxruntime" / "quantization" / "operators" / "conv.py"

    patches_str = str(patches.resolve())
    if patches_str not in sys.path:
        sys.path.insert(0, patches_str)

    # 先拿系统 ORT，再覆盖子模块（与旧逻辑一致）
    was_in_path = False
    if patches_str in sys.path:
        sys.path.remove(patches_str)
        was_in_path = True

    try:
        import onnxruntime as system_ort  # noqa: WPS433
        from onnxruntime.quantization import operators as system_operators  # noqa: WPS433

        if "onnxruntime" not in sys.modules:
            sys.modules["onnxruntime"] = system_ort
        if "onnxruntime.quantization" not in sys.modules:
            from onnxruntime import quantization as system_quantization  # noqa: WPS433

            sys.modules["onnxruntime.quantization"] = system_quantization

        # 父包可能已把子模块绑到属性上；仅写 sys.modules 不够，需同步替换属性
        quant_pkg = sys.modules.get("onnxruntime.quantization")
        ops_pkg = sys.modules.get("onnxruntime.quantization.operators", system_operators)

        if quant_utils_path.is_file():
            spec = importlib.util.spec_from_file_location(
                "onnxruntime.quantization.quant_utils",
                str(quant_utils_path),
            )
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules["onnxruntime.quantization.quant_utils"] = mod
                spec.loader.exec_module(mod)
                if quant_pkg is not None:
                    setattr(quant_pkg, "quant_utils", mod)
                print(f"✓ 已加载本地 quant_utils.py: {quant_utils_path}")
            else:
                print(f"⚠ 无法加载: {quant_utils_path}")
        else:
            print(f"⚠ 缺少补丁文件: {quant_utils_path}")

        if conv_path.is_file():
            if "onnxruntime.quantization.operators" not in sys.modules:
                sys.modules["onnxruntime.quantization.operators"] = system_operators
                ops_pkg = system_operators
            spec = importlib.util.spec_from_file_location(
                "onnxruntime.quantization.operators.conv",
                str(conv_path),
            )
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules["onnxruntime.quantization.operators.conv"] = mod
                spec.loader.exec_module(mod)
                if ops_pkg is not None:
                    setattr(ops_pkg, "conv", mod)
                if quant_pkg is not None and hasattr(quant_pkg, "operators"):
                    # 保持 quant_pkg.operators.conv 一致
                    setattr(quant_pkg.operators, "conv", mod)
                print(f"✓ 已加载本地 conv.py: {conv_path}")
            else:
                print(f"⚠ 无法加载: {conv_path}")
        else:
            print(f"⚠ 缺少补丁文件: {conv_path}")
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"加载本地 onnxruntime 补丁失败: {e}", UserWarning)
    finally:
        if was_in_path and patches_str not in sys.path:
            sys.path.insert(0, patches_str)


if __name__ != "__main__":
    load_local_onnxruntime_modules()
