# 静态量化脚本
# 调用onnxruntime工具预处理onnx模型，为量化做准备
import time
import psutil
import gc
import sys
import os

# 本地 package「platform/」会遮蔽标准库；须在 torch/ultralytics 之前固定 stdlib
import _pin_stdlib_platform  # noqa: F401, E402

from ultralytics import YOLO
# 对比误差
import cv2
import matplotlib.pyplot as plt
import onnx


# 强制使用本地版本的 quant_utils.py 和 conv.py（如果存在）
# 这个模块会自动设置路径和加载本地版本
import _load_local_onnxruntime_modules  # noqa: E402

# 导入 onnxruntime 基础模块
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader
from onnxruntime.quantization import quantize_static, QuantType, QuantFormat

import re
from onnx import helper
import numpy as np
print(f"onnxruntime 模块路径: {ort.__file__}")

from grayscale_preprocess import PREPROCESS_MODES, preprocess_by_mode  # noqa: E402
from typing import List, Optional, Union  # noqa: E402

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _resolve_path(p: Union[str, os.PathLike]) -> str:
    path = os.path.abspath(os.path.expanduser(str(p)))
    return path


def resolve_calibration_folder(current_folder: str, cali_dir: Optional[str] = None) -> str:
    """解析校准图目录：显式参数 > 环境变量 > job 布局 > 引擎旁遗留 cali_data。"""
    candidates = []  # type: List[str]
    if cali_dir:
        candidates.append(_resolve_path(cali_dir))
    env = os.environ.get("QUANTITIZE_CALI_DIR", "").strip()
    if env:
        candidates.append(_resolve_path(env))
    ws = _resolve_path(current_folder)
    candidates.append(os.path.join(os.path.dirname(ws), "input", "cali"))
    candidates.append(os.path.join(ws, "cali"))
    candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cali_data"))

    tried = []
    for folder in candidates:
        if folder in tried:
            continue
        tried.append(folder)
        if not os.path.isdir(folder):
            continue
        has_img = any(
            f.lower().endswith(_IMAGE_EXTS) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))
        )
        if has_img:
            print(f"✓ 使用校准目录: {folder}")
            return folder

    raise FileNotFoundError(
        "未找到可用的校准图片目录。已尝试:\n  - "
        + "\n  - ".join(tried)
        + "\n请用 --cali-dir 指定，或设置 QUANTITIZE_CALI_DIR，"
        "或使用标准 job 布局（workspace 的同级 input/cali）。"
    )


def create_calibration_paths(calibration_folder, current_folder):
    txt_folder = current_folder
    output_file = os.path.join(txt_folder, 'calibration_images.txt')

    # 获取图片文件列表
    image_files = sorted([
        os.path.join(calibration_folder, f)
        for f in os.listdir(calibration_folder)
        if f.lower().endswith(_IMAGE_EXTS) and os.path.isfile(os.path.join(calibration_folder, f))
    ])

    # 确保输出目录存在
    os.makedirs(txt_folder, exist_ok=True)

    # 写入文件
    with open(output_file, 'w') as f:
        for img_path in image_files:
            f.write(f"{img_path},\n")
    print(f"成功保存 {len(image_files)} 个图片路径到 {output_file}")
    return output_file

def load_calibration_paths(file_path):
    """
    从文本文件加载校准图像路径列表

    参数:
        file_path (str): 包含图像路径的文本文件路径

    返回:
        list: 校准图像路径列表
    """
    calibration_paths = []

    try:
        with open(file_path, 'r') as file:
            for line in file:
                # 移除行尾的换行符和逗号
                line = line.strip().rstrip(',')
                # 如果行不为空则添加到列表
                if line:
                    calibration_paths.append(line)
        return calibration_paths
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return []  # 出错时返回空列表




class PoseCalibrationDataReader(CalibrationDataReader):
    def __init__(self, image_paths, preprocess_mode: str = "grayscale_uniform"):
        self.image_paths = image_paths
        self.idx = 0
        self.input_name = "images"  # YOLOv8 标准输入名称
        mode = (preprocess_mode or "grayscale_uniform").strip()
        if mode not in PREPROCESS_MODES:
            raise ValueError(f"未知 preprocess_mode: {mode!r}，可选 {PREPROCESS_MODES}")
        self.preprocess_mode = mode

    def preprocess(self, img_path):
        try:
            return preprocess_by_mode(img_path, self.preprocess_mode, target_size=(1280, 1280), dtype=np.float16)
        except Exception as e:
            print(f"警告: 无法预处理图像 {img_path}: {e}")
            return None

    def get_next(self):
        if self.idx >= len(self.image_paths):
            return None

        img_path = self.image_paths[self.idx]
        self.idx += 1

        img_data = self.preprocess(img_path)
        if img_data is None:
            # 如果图像读取失败，尝试下一个
            return self.get_next()

        return {self.input_name: img_data}




def print_memory_usage(stage=""):
    """打印当前内存使用情况"""
    memory = psutil.virtual_memory()
    process = psutil.Process()
    print(f"\n=== 内存使用情况 - {stage} ===")
    print(f"系统总内存: {memory.total / (1024**3):.1f} GB")
    print(f"系统已用内存: {memory.used / (1024**3):.1f} GB")
    print(f"系统可用内存: {memory.available / (1024**3):.1f} GB")
    print(f"系统内存使用率: {memory.percent:.1f}%")
    print(f"当前进程内存: {process.memory_info().rss / (1024**2):.1f} MB")
    print(f"当前进程内存百分比: {process.memory_percent():.2f}%")




def get_conv_output_shapes_list(model_path):
    """获取ONNX模型中所有卷积层输出shape的list"""

    model = onnx.load(model_path)

    # 创建值信息字典
    value_info = {}
    for info in model.graph.value_info:
        value_info[info.name] = info

    # 存储所有卷积层输出shape
    conv_shapes = []
    conv_info = []

    # 遍历所有节点
    for i, node in enumerate(model.graph.node):
        if node.op_type == 'Add' or node.op_type == 'Conv' or node.op_type == 'Mul'or \
            node.op_type == 'Split' or node.op_type == 'Concat' or node.op_type == 'Slice' \
            or node.op_type=='MaxPool' or node.op_type=='Resize' or node.op_type=='Reshape' or node.op_type=='Sigmoid'\
            or node.op_type=='Sub' or node.op_type=='Transpose' or node.op_type=='Softmax':
            # 获取输出shape
            for output in node.output:
                if output in value_info:
                    output_info = value_info[output]
                    shape = [dim.dim_value if dim.dim_value > 0 else dim.dim_param for dim in output_info.type.tensor_type.shape.dim]
                    conv_shapes.append(shape)
                    conv_info.append({
                        'index': i,
                        'name': node.name,
                        'output_name': output,
                        'shape': shape
                    })
                else:
                    conv_shapes.append(None)
                    conv_info.append({
                        'index': i,
                        'name': node.name,
                        'output_name': node.output[0],
                        'shape': None
                    })
    print(len(conv_shapes), len(conv_info))
    return conv_shapes, conv_info

def add_first_conv_output(model_path, output_path):
    """只添加第一个Conv层的输出，使用指定形状"""

    # 加载原始模型
    model = onnx.load(model_path)
    nodes = model.graph.node

    print("=== 添加第一个Conv层输出 ===")


    # 创建中间输出
    intermediate_outputs = []
       # 获取所有Conv层的shape信息
    conv_shapes, conv_info = get_conv_output_shapes_list(model_path)
    # 为第一个Conv的每个输出创建输出信息
    for i, (shape, info) in enumerate(zip(conv_shapes, conv_info)):

        # output_info = helper.make_tensor_value_info(
        #     info['output_name'],
        #     onnx.TensorProto.FLOAT16,
        #     shape
        # )
        
        output_info = helper.make_tensor_value_info(
            info['output_name'],
            onnx.TensorProto.FLOAT16,
            shape
        )
        
        intermediate_outputs.append(output_info)

    # 添加中间输出到模型
    model.graph.output.extend(intermediate_outputs)

    # 保存修改后的模型
    onnx.save(model, output_path)
    print(f"修改后的模型已保存到: {output_path}")

    return output_path





def extract_excluded_node_types(model_path, output_file):
    """
    提取需要排除激活值量化的节点类型
    (只排除非卷积节点类型，保留卷积节点的激活值量化)
    """
    model = onnx.load(model_path)

    # 获取模型中所有唯一的节点类型
    all_node_types = set(node.op_type for node in model.graph.node)
    print(f"模型中共有 {len(all_node_types)} 种不同的节点类型")
    print("所有节点类型:", sorted(all_node_types))

    # 定义需要排除的节点类型
    excluded_types = set()

    # 添加所有非卷积节点类型
    conv_ops = {"Conv", "ConvTranspose", "QLinearConv", "ConvInteger"}
    for node in model.graph.node:
        if node.op_type not in conv_ops:
            excluded_types.add(node.op_type)

    # 不再添加卷积节点类型（保留卷积节点的激活值量化）
    # excluded_types.update(conv_ops)  # 注释掉这行

    # 添加特定的量化操作节点类型
    sensitive_ops = {"QuantizeLinear", "DequantizeLinear"}
    excluded_types.update(sensitive_ops)

    # 计算剩余节点类型（需要量化的节点类型）
    remaining_types = all_node_types - excluded_types

    # 打印详细的统计信息
    print("\n===== 节点类型统计 =====")
    print(f"排除的节点类型数量: {len(excluded_types)}")
    print("排除的节点类型:", sorted(excluded_types))
    print(f"\n保留的节点类型数量: {len(remaining_types)}")
    print("保留的节点类型:", sorted(remaining_types))

    # 将结果写入文本文件
    with open(output_file, 'w') as f:
        for node_type in sorted(excluded_types):
            f.write(node_type + '\n')

    print(f"\n已导出 {len(excluded_types)} 种需要排除的节点类型至: {output_file}")


def extract_node(output_file, model_path, excluded_nodes_file):



    def read_excluded_node_types(file_path):
        """读取需要排除的节点类型列表"""
        with open(file_path, 'r') as f:
            return set(line.strip() for line in f)

    def get_nodes_to_exclude(model_path, excluded_types):
        """获取需要排除的具体节点名称"""
        model = onnx.load(model_path)
        excluded_nodes = []

        print(f"\n模型包含 {len(model.graph.node)} 个节点")
        print(f"需要排除的节点类型: {excluded_types}")

        # 统计节点类型分布
        type_count = {}
        for node in model.graph.node:
            if node.op_type not in type_count:
                type_count[node.op_type] = 0
            type_count[node.op_type] += 1

        print("\n节点类型分布统计:")
        for op_type, count in sorted(type_count.items()):
            print(f" - {op_type}: {count}个节点")

        # 收集所有匹配的节点
        for node in model.graph.node:
            if node.op_type in excluded_types:
                excluded_nodes.append(node.name)

        # 添加额外的排除规则：包含特定关键字的节点
        additional_excluded = []
        for node in model.graph.node:
            # 包含"act"、"Sigmoid"、"Mul"的节点
            if "act" in node.name or "Sigmoid" in node.name or "Mul" in node.name or 'Concat' in node.name:
                if node.name not in excluded_nodes:
                    additional_excluded.append(node.name)

        # 合并列表
        all_excluded = excluded_nodes + additional_excluded

        # 统计排除的节点类型
        excluded_type_count = {}
        for node in model.graph.node:
            if node.name in all_excluded:
                if node.op_type not in excluded_type_count:
                    excluded_type_count[node.op_type] = 0
                excluded_type_count[node.op_type] += 1

        print("\n===== 排除节点统计 =====")
        print(f"基于类型找到 {len(excluded_nodes)} 个排除节点")
        print(f"基于关键字找到 {len(additional_excluded)} 个额外排除节点")
        print(f"总计排除节点数: {len(all_excluded)}")

        print("\n排除节点的类型分布:")
        for op_type, count in sorted(excluded_type_count.items()):
            print(f" - {op_type}: {count}个节点")

        # 计算保留的节点
        retained_nodes = [node.name for node in model.graph.node if node.name not in all_excluded]
        retained_type_count = {}
        for node in model.graph.node:
            if node.name in retained_nodes:
                if node.op_type not in retained_type_count:
                    retained_type_count[node.op_type] = 0
                retained_type_count[node.op_type] += 1

        print("\n===== 保留节点统计 =====")
        print(f"保留节点总数: {len(retained_nodes)}")
        print("保留节点的类型分布:")
        for op_type, count in sorted(retained_type_count.items()):
            print(f" - {op_type}: {count}个节点")

        return all_excluded

    def save_excluded_nodes_to_file(nodes, output_path):
        """将排除节点名称保存到文本文件"""
        with open(output_path, 'w') as f:
            for node_name in nodes:
                f.write(f"{node_name}\n")
        print(f"已保存 {len(nodes)} 个排除节点到: {os.path.abspath(output_path)}")

    # 设置输出目录


    # 读取需要排除的节点类型
    excluded_types = read_excluded_node_types(output_file)
    print(f"从文件读取到 {len(excluded_types)} 种需要排除的节点类型")
    print("排除类型列表:", sorted(excluded_types))

    # 获取需要排除的具体节点名称
    excluded_nodes = get_nodes_to_exclude(model_path, excluded_types)

    # 保存排除节点名称到文件
   
    save_excluded_nodes_to_file(excluded_nodes, excluded_nodes_file)

    print(f"将排除 {len(excluded_nodes)} 个节点的激活值量化")

    # 打印前10个排除节点作为示例
    print("\n排除节点示例:")
    for node in excluded_nodes[:10]:
        print(f" - {node}")

    # 打印关键统计信息
    conv_nodes = [n for n in excluded_nodes if "Conv" in n]
    print(f"\n排除列表中包含 {len(conv_nodes)} 个卷积节点")
    if conv_nodes:
        print("卷积节点示例:", conv_nodes[:3])



def load_and_preprocess_real_image(image_path, target_size=(1280, 1280)):
    """加载和预处理真实图像"""

    print(f"加载图像: {image_path}")

    # 方法1: 使用OpenCV加载
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        raise ValueError(f"无法加载图像: {image_path}")

    # 转换BGR到RGB
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

    # 调整图像尺寸
    image_resized = cv2.resize(image_rgb, target_size)

    # 归一化到[0,1]
    # image_normalized = image_resized.astype(np.float16) / 255.0
    image_normalized = image_resized.astype(np.float32) / 255.0
    image_normalized = image_normalized.astype(np.float16)
    # 转换为CHW格式 (H, W, C) -> (C, H, W)
    image_chw = np.transpose(image_normalized, (2, 0, 1))

    # 添加batch维度 (C, H, W) -> (1, C, H, W)
    image_batch = np.expand_dims(image_chw, axis=0)

    print(f"图像形状: {image_batch.shape}")
    print(f"图像数据类型: {image_batch.dtype}")
    print(f"图像值范围: [{np.min(image_batch):.4f}, {np.max(image_batch):.4f}]")
    return image_batch


def plot_layer_mse_comparison_fixed(mse_data, mae_data, rel_error_data, layer_names, onnx_name, current_folder):
    """绘制每层的MSE对比图（修复中文字体问题）"""

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Quantization Model Layer-wise Error Analysis', fontsize=16, fontweight='bold')

    # 子图1: MSE对比
    axes[0, 0].plot(range(len(mse_data)), mse_data, 'b-o', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Layer Index')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].set_title('Layer-wise MSE Comparison')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')

    # 子图2: MAE对比
    axes[0, 1].plot(range(len(mae_data)), mae_data, 'r-s', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Layer Index')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title('Layer-wise MAE Comparison')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')

    # 子图3: 相对误差对比
    axes[1, 0].plot(range(len(rel_error_data)), rel_error_data, 'g-^', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Layer Index')
    axes[1, 0].set_ylabel('Relative Error (%)')
    axes[1, 0].set_title('Layer-wise Relative Error Comparison')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')

    # 子图4: 综合对比 (归一化)；用 nanmax 避免 inf/nan 导致无效除法
    def _safe_normalize(data):
        arr = np.asarray(data, dtype=np.float64)
        peak = np.nanmax(arr[np.isfinite(arr)]) if np.any(np.isfinite(arr)) else 0.0
        return arr / peak if peak > 0 else arr

    mse_normalized = _safe_normalize(mse_data)
    mae_normalized = _safe_normalize(mae_data)
    rel_error_normalized = _safe_normalize(rel_error_data)

    axes[1, 1].plot(range(len(mse_normalized)), mse_normalized, 'b-o', label='MSE (Normalized)', linewidth=2, markersize=6)
    axes[1, 1].plot(range(len(mae_normalized)), mae_normalized, 'r-s', label='MAE (Normalized)', linewidth=2, markersize=6)
    axes[1, 1].plot(range(len(rel_error_normalized)), rel_error_normalized, 'g-^', label='Relative Error (Normalized)', linewidth=2, markersize=6)
    axes[1, 1].set_xlabel('Layer Index')
    axes[1, 1].set_ylabel('Normalized Error')
    axes[1, 1].set_title('Comprehensive Error Comparison (Normalized)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 调整布局
    plt.tight_layout()

    # 保存图表
    plt.savefig(f'{current_folder}/{onnx_name}_layer_mse_comparison.png',
                dpi=300, bbox_inches='tight')

def compare_corresponding_layers(original_path, quantized_path, input_data, info, onnx_name, current_folder):
    """按输出张量名对齐后对比对应层（量化会改变节点遍历顺序，不能按索引硬对齐）。"""
    original_session = ort.InferenceSession(original_path)
    quantized_session = ort.InferenceSession(quantized_path)

    original_results = original_session.run(None, input_data)
    quantized_results = quantized_session.run(None, input_data)

    orig_names = [o.name for o in original_session.get_outputs()]
    quant_names = [o.name for o in quantized_session.get_outputs()]

    # 同名输出可能重复（如 output0）；保留首次出现
    orig_by_name = {}
    for name, value in zip(orig_names, original_results):
        if name not in orig_by_name:
            orig_by_name[name] = value
    quant_by_name = {}
    for name, value in zip(quant_names, quantized_results):
        if name not in quant_by_name:
            quant_by_name[name] = value

    info_by_output = {item['output_name']: item for item in info}
    common_names = [n for n in orig_by_name if n in quant_by_name]
    # 按原始模型输出顺序遍历，保证曲线稳定
    ordered_names = [n for n in orig_names if n in quant_by_name]
    # 去重保序
    seen = set()
    ordered_names = [n for n in ordered_names if not (n in seen or seen.add(n))]

    print("\n=== 对应层对比结果 ===")
    print(f"原始输出: {len(orig_by_name)}, 量化输出: {len(quant_by_name)}, 按名称对齐: {len(common_names)}")

    layer_mse_data = []
    layer_mae_data = []
    layer_rel_error_data = []
    layer_names = []
    skipped = 0

    for i, name in enumerate(ordered_names):
        orig_output = np.asarray(orig_by_name[name], dtype=np.float32)
        quant_output = np.asarray(quant_by_name[name], dtype=np.float32)

        if orig_output.shape != quant_output.shape:
            skipped += 1
            print(f"警告: 输出形状不匹配，无法直接对比: {name} "
                  f"{orig_output.shape} vs {quant_output.shape}")
            continue

        # float32 计算，避免 fp16 平方溢出
        diff = orig_output - quant_output
        meta = info_by_output.get(name)
        print(i, meta if meta is not None else {'output_name': name}, float(diff.min()), float(diff.max()))

        mse = float(np.mean(diff ** 2))
        mae = float(np.mean(np.abs(diff)))
        rel_error = float(np.mean(np.abs(diff) / (np.abs(orig_output) + 1e-6)) * 100)

        layer_mse_data.append(mse)
        layer_mae_data.append(mae)
        layer_rel_error_data.append(rel_error)
        if name == 'output0' or meta is None:
            layer_names.append('total_output' if name == 'output0' else name)
        else:
            layer_names.append(meta['name'])

    if skipped:
        print(f"跳过形状不匹配层数: {skipped}")

    # 把 final output 放到末尾，便于观察逐层误差积累
    if len(layer_mse_data) > 1 and layer_names[0] == 'total_output':
        layer_mse_data = layer_mse_data[1:] + layer_mse_data[:1]
        layer_mae_data = layer_mae_data[1:] + layer_mae_data[:1]
        layer_rel_error_data = layer_rel_error_data[1:] + layer_rel_error_data[:1]
        layer_names = layer_names[1:] + layer_names[:1]

    plot_layer_mse_comparison_fixed(
        layer_mse_data, layer_mae_data, layer_rel_error_data, layer_names, onnx_name, current_folder
    )
    return original_results, quantized_results


def quantize_static_inner(excluded_nodes_file, calibration_data_reader, model_output, model_input):   
    from onnxruntime.quantization import QuantType, QuantFormat, quantize_static

    from onnxruntime.quantization import quantize_static, QuantType, QuantFormat

    # 姿态模型需要排除的关键节点 (根据网络结构调整)
    with open(excluded_nodes_file, 'r') as f:
        excluded_nodes = f.read().splitlines()
    # excluded_nodes += ['/model.4/m.5/cv2/conv/Conv']
    # import ipdb;ipdb.set_trace()
    tensor_quant_overrides = {
        'images': [{'quant_type': None}]  # 不量化images张量
    }
    extra_options = {
        'CalibrateMethod': 'Entropy',
        'num_bins': 1024,  # 进一步增加到1024
        'num_quantized_bins': 1024,
        'OpTypesToExcludeOutputQuantization': ['Mul','Conv', 'MatMul', 'Gemm', 'Add', 'Sigmoid', 'Relu'],
        'QuantizeBias': False 
    }
    # 在量化前打印内存状态
    print_memory_usage("Entropy")

    start_time = time.time()
    import sys

    # 强制垃圾回收
    gc.collect()
    from onnxruntime.quantization import CalibrationMethod
    print_memory_usage("量化开始前（垃圾回收后）")
    if not os.path.exists(model_output):
        print(f'{model_output}')
        # ORT 1.x hard-codes calibration inference to CPU even when the CUDA
        # provider is installed.  The public calibrator exposes a provider
        # setter but quantize_static does not forward it, so patch only the
        # factory in this single-task process and restore it afterward.
        provider_mode = os.environ.get("QUANTIZE_CALIBRATION_PROVIDER", "cpu").strip().lower()
        quantize_globals = quantize_static.__globals__
        original_create_calibrator = quantize_globals.get("create_calibrator")

        if provider_mode == "cuda" and original_create_calibrator is not None:
            def _create_cuda_calibrator(*args, **kwargs):
                calibrator = original_create_calibrator(*args, **kwargs)
                calibrator.set_execution_providers(
                    ["CUDAExecutionProvider", "CPUExecutionProvider"]
                )
                print("✓ 量化校准 providers: CUDAExecutionProvider, CPUExecutionProvider")
                return calibrator

            quantize_globals["create_calibrator"] = _create_cuda_calibrator
        else:
            print("量化校准 provider: CPUExecutionProvider")
        try:
            quantize_static(
                model_input=model_input,
                model_output=model_output,
                calibration_data_reader=calibration_data_reader,
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QInt16,  # 激活值采用16位
                weight_type=QuantType.QInt8,
                per_channel=True,
                reduce_range=False,
                nodes_to_exclude=excluded_nodes,  # 排除敏感节点
                extra_options=extra_options,
            )
        finally:
            if original_create_calibrator is not None:
                quantize_globals["create_calibrator"] = original_create_calibrator
    end_time = time.time()
    quantization_time = end_time - start_time
    print(f"量化过程耗时: {quantization_time/60:.2f} 分钟")

    # 量化后打印内存状态
    print_memory_usage("量化完成后")
    #量化时间：6min25s

def load_calibration_folder(current_folder, cali_dir=None, preprocess_mode="grayscale_uniform"):
    calibration_folder = resolve_calibration_folder(current_folder, cali_dir=cali_dir)

    # 创建校准图片路径列表文件（保存在 workspace/info_txt）
    info_txt_folder = os.path.join(current_folder, 'info_txt')
    os.makedirs(info_txt_folder, exist_ok=True)
    calibration_images_txt = os.path.join(info_txt_folder, 'calibration_images.txt')
    print(calibration_folder, "-----------------------------------------------------")
    create_calibration_paths(calibration_folder, info_txt_folder)

    calibration_image_paths = load_calibration_paths(calibration_images_txt)

    if not calibration_image_paths:
        raise ValueError(
            f"未找到校准图片。请确保 {calibration_folder} 文件夹中包含图片文件"
        )

    print(f"✓ 校准预处理模式: {preprocess_mode}（共 {len(calibration_image_paths)} 张）")
    return PoseCalibrationDataReader(calibration_image_paths, preprocess_mode=preprocess_mode)
def convert_pt_to_onnx_fp16(pt_model_path, output_onnx_path, imgsz=1280, dynamic=False, opset=17, device='0'):
    """
    将 PyTorch 模型 (*.pt) 转换为 ONNX 格式 (FP16精度)
    与 export_onnx_with_gpu.py 中的 export_with_ultralytics 函数对齐
    
    Args:
        pt_model_path: PyTorch 模型路径 (*.pt)
        output_onnx_path: 输出的 ONNX 文件路径 (*.onnx)
        imgsz: 导出时的输入尺寸 (正方形)，默认 1280
        dynamic: 是否启用动态轴，默认 False
        opset: ONNX opset 版本，默认 17
        device: GPU 设备，例如 '0' 或 'cuda:0'，默认 '0'
    
    Returns:
        str: 转换后的 ONNX 文件路径，如果失败返回 None
    """
    try:
        print(f"正在将 PyTorch 模型转换为 ONNX (FP16): {pt_model_path}")
        
        # 检查输入文件是否存在
        if not os.path.exists(pt_model_path):
            raise FileNotFoundError(f"PyTorch 模型文件不存在: {pt_model_path}")
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_onnx_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # 检查 CUDA 是否可用
        try:
            import torch
            if not torch.cuda.is_available():
                print("⚠ 警告: CUDA 不可用，将使用 CPU 导出（可能较慢）")
        except ImportError:
            print("⚠ 警告: PyTorch 未安装，无法检查 CUDA 可用性")
        
        # 处理 device 参数：兼容数字设备参数
        dev = device
        if isinstance(device, str) and device.isdigit():
            dev = int(device)
        
        # 使用 ultralytics YOLO 加载模型
        model = YOLO(pt_model_path)
        
        # 导出为 ONNX 格式，使用 FP16 精度
        # 与 export_onnx_with_gpu.py 中的参数对齐
        print(f"  导出参数: format=onnx, half=True (FP16), imgsz={imgsz}, dynamic={dynamic}, opset={opset}, device={dev}")
        exported_path = model.export(
            format='onnx',
            half=True,  # 使用 FP16 精度
            opset=opset,  # ONNX opset 版本
            imgsz=imgsz,  # 输入尺寸
            dynamic=dynamic,  # 是否启用动态轴
            device=dev,  # GPU 设备
        )
        
        # model.export() 返回的路径可能不是我们指定的路径
        # 如果 exported_path 与目标路径不同，需要移动文件
        if exported_path != output_onnx_path and os.path.exists(exported_path):
            try:
                # 优先使用 os.replace（原子操作）
                os.replace(exported_path, output_onnx_path)
                print(f"  文件已移动: {exported_path} -> {output_onnx_path}")
            except Exception:
                # 如果 replace 失败，使用 shutil.copyfile
                import shutil
                shutil.copyfile(exported_path, output_onnx_path)
                try:
                    os.remove(exported_path)
                except Exception:
                    pass
                print(f"  文件已复制: {exported_path} -> {output_onnx_path}")
        
        if os.path.exists(output_onnx_path):
            file_size = os.path.getsize(output_onnx_path) / (1024 * 1024)  # MB
            print(f"✓ 转换成功: {output_onnx_path} ({file_size:.2f} MB)")
            return output_onnx_path
        else:
            print(f"⚠ 警告: 转换后的文件不存在: {output_onnx_path}")
            if os.path.exists(exported_path):
                print(f"  但找到了文件: {exported_path}")
                return exported_path
            return None
            
    except Exception as e:
        print(f"✗ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        return None



if __name__ == '__main__':
    # 参数顺序：folder_path, onnx_name, model_path (与 process_pipeline.py 保持一致)
    import argparse

    parser = argparse.ArgumentParser(description="PT→FP16 ONNX 静态量化")
    parser.add_argument("folder_path", help="workspace 目录")
    parser.add_argument("onnx_name", help="ONNX 基名（不含扩展名）")
    parser.add_argument("model_path", help="原始 *.pt 路径")
    parser.add_argument("--cali-dir", default=None, help="校准图目录（可选）")
    parser.add_argument("--preprocess-mode", default=None, help="预处理模式（可选）")
    args = parser.parse_args()

    current_folder = args.folder_path
    onnx_name = args.onnx_name
    pt_model_path = args.model_path
    cali_dir = args.cali_dir
    preprocess_mode = (args.preprocess_mode or "grayscale_uniform").strip()
    if preprocess_mode not in PREPROCESS_MODES:
        print(f"错误: 未知 --preprocess-mode={preprocess_mode!r}，可选 {PREPROCESS_MODES}")
        sys.exit(2)

    
    # 将 *.pt 模型转换为 *.onnx (FP16)
    # 转换后的 ONNX 文件路径
    model_path = f'{current_folder}/{onnx_name}_fp16.onnx'
    
    # 检查是否已经存在转换后的 ONNX 文件
    if not os.path.exists(model_path):
        print(f"\n{'='*70}")
        print("步骤 1: 将 PyTorch 模型转换为 ONNX (FP16)")
        print(f"{'='*70}")
        # 使用与 export_onnx_with_gpu.py 相同的默认参数
        converted_path = convert_pt_to_onnx_fp16(
            pt_model_path, 
            model_path,
            imgsz=1280,  # 默认输入尺寸
            dynamic=False,  # 默认不启用动态轴
            opset=17,  # 默认 opset 17（与 export_onnx_with_gpu.py 对齐）
            device='0'  # 默认使用 GPU 0
        )
        if converted_path is None:
            print("错误: PyTorch 模型转换为 ONNX 失败")
            sys.exit(1)
        model_path = converted_path
    else:
        print(f"✓ ONNX 文件已存在，跳过转换: {model_path}")
    
    model_output = f'{current_folder}/{onnx_name}.onnx'
    # 指定模型路径和输出文件
    output_file = f"{current_folder}/{os.path.basename(model_path).replace('.onnx', 'excluded_node_types.txt')}"
    if not os.path.exists(output_file):
        extract_excluded_node_types(model_path, output_file)
    excluded_nodes_file = os.path.join(output_file.replace('excluded_node_types.txt', 'excluded_nodes.txt'))
    if not os.path.exists(excluded_nodes_file):
        extract_node(output_file, model_path, excluded_nodes_file)

    calibration_data_reader = load_calibration_folder(
        current_folder,
        cali_dir=cali_dir,
        preprocess_mode=preprocess_mode,
    )
   
    quantize_static_inner(excluded_nodes_file, calibration_data_reader, model_output, model_path)


    # # 使用修改后的函数


    original_path = f'{current_folder}/' + os.path.basename(model_path).replace('.onnx', '_output.onnx')
    if not os.path.exists(original_path):
            add_first_conv_output(model_path, original_path)

    model_path_quantized = f'{current_folder}/{onnx_name}.onnx'
    output_path = f'{current_folder}/{onnx_name}_output.onnx'
    if not os.path.exists(output_path):
        add_first_conv_output(model_path_quantized, output_path)


    # 层误差对比：从校准目录取第一张图，避免写死文件名
    cali_resolved = resolve_calibration_folder(current_folder, cali_dir=cali_dir)
    sample_images = sorted(
        os.path.join(cali_resolved, f)
        for f in os.listdir(cali_resolved)
        if f.lower().endswith(_IMAGE_EXTS) and os.path.isfile(os.path.join(cali_resolved, f))
    )
    if not sample_images:
        print("⚠ 校准目录无图片，跳过层误差对比")
        sys.exit(0)
    image_path = sample_images[0]
    print(f"✓ 层对比样例图: {image_path}")

    image = preprocess_by_mode(image_path, preprocess_mode, target_size=(1280, 1280), dtype=np.float16)
    input_data = {'images': image}

    quantized_model_path = f'{current_folder}/{onnx_name}_output.onnx'
    _, info= get_conv_output_shapes_list(original_path)
    original_results, quantized_results = compare_corresponding_layers(
        original_path,
        quantized_model_path,
        input_data,
        info,
        onnx_name,
        current_folder
    )
    print('32')
