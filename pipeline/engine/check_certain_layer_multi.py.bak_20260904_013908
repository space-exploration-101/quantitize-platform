#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导出特定层的输入输出数据
专门用于导出 /model.4/m.5/cv1/conv/Conv_output_0 这一层的数据
修正版本：确保同时导出和打印输入输出数据
"""

import sys
import os

# 首先清理 sys.path，移除 Isaac Sim 等可能冲突的路径
# 这必须在导入任何其他模块之前完成
sys.path = [p for p in sys.path if p and 'isaac-sim' not in p.lower()]

import _pin_stdlib_platform  # noqa: F401, E402

from re import M
import numpy as np
import onnx
from onnx import helper
from onnx.numpy_helper import to_array as onnx_tensor_to_array

# 设置本地 onnxruntime 路径（必须在导入 onnxruntime 之前）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _setup_local_onnxruntime  # noqa: E402

import onnxruntime as ort
import cv2

def load_and_preprocess_real_image(image_path, target_size=(1280, 1280)):
    """加载和预处理真实图像"""
    print(f"加载图像: {image_path}")

    # 使用OpenCV加载
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        raise ValueError(f"无法加载图像: {image_path}")

    # 转为灰度图（黑白）
    image_gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    # 调整图像尺寸
    image_resized = cv2.resize(image_gray, target_size)

    # 归一化到[0,1]
    image_normalized = image_resized.astype(np.float16) / 255.0
    # 灰度图 (H, W) 复制为三通道，每通道值相同 -> (3, H, W) CHW 格式
    image_chw = np.stack([image_normalized, image_normalized, image_normalized], axis=0)

    # 添加 batch 维度 (C, H, W) -> (1, C, H, W)
    image_batch = np.expand_dims(image_chw, axis=0)

    print(f"图像形状: {image_batch.shape}")
    print(f"图像数据类型: {image_batch.dtype}")
    print(f"图像值范围: [{np.min(image_batch):.4f}, {np.max(image_batch):.4f}]")
    return image_batch
def get_initializer_data(init):
    """
    获取ONNX初始值的数据，支持多种存储方式
    """
    import numpy as np
    
    # 检查数据类型
    if init.data_type == 1:  # FLOAT
        if init.raw_data:
            return np.frombuffer(init.raw_data, dtype=np.float32)
        elif init.float_data:
            return np.array(init.float_data, dtype=np.float32)
    elif init.data_type == 2:  # UINT8
        if init.raw_data:
            return np.frombuffer(init.raw_data, dtype=np.uint8)
        elif init.int32_data:
            return np.array(init.int32_data, dtype=np.uint8)
    elif init.data_type == 3:  # INT8
        if init.raw_data:
            return np.frombuffer(init.raw_data, dtype=np.int8)
        elif init.int32_data:
            return np.array(init.int32_data, dtype=np.int8)
    elif init.data_type == 6:  # INT32
        if init.raw_data:
            return np.frombuffer(init.raw_data, dtype=np.int32)
        elif init.int32_data:
            return np.array(init.int32_data, dtype=np.int32)
    elif init.data_type == 7:  # INT64
        if init.raw_data:
            return np.frombuffer(init.raw_data, dtype=np.int64)
        elif init.int64_data:
            return np.array(init.int64_data, dtype=np.int64)
    elif init.data_type == 10:  # FLOAT16
        if init.int32_data:
            # 非标准方式：从int32_data读取
            print(f"从int32_data读取BFLOAT16: {init.name}")
            int32_array = np.array(init.int32_data, dtype=np.uint32)
            
            # 尝试两种方式
            try:
                # 方式1: 高16位
                bfloat16_high = (int32_array >> 16).astype(np.float16)
                print(f"高16位数据: {bfloat16_high[:5]}...")
                # 方式2: 低16位
                bfloat16_low = (int32_array & 0xFFFF).astype(np.uint16)
                print(f"低16位数据: {bfloat16_low[:5]}...")
                
                # 选择看起来更合理的数据
                # 通常BFLOAT16数据不会全为0或全为1
                if np.any(bfloat16_high != 0) and np.any(bfloat16_high != 0xFFFF):
                    print("选择高16位数据")
                    return bfloat16_high
                elif np.any(bfloat16_low != 0) and np.any(bfloat16_low != 0xFFFF):
                    print("选择低16位数据")
                    return bfloat16_low
                else:
                    print("无法确定BFLOAT16数据位置")
                    return None
                    
            except Exception as e:
                print(f"转换BFLOAT16时出错: {e}")
                return None
        elif init.float_data:
            return np.array(init.float_data, dtype=np.float16)
    
    return None
def get_quantized_operator_outputs(model_path, target_name, input_data, input_shape=None):

    """
    根据给定名字获取量化算子的输出
    
    参数:
        model_path: ONNX模型文件路径
        target_name: 目标算子输出名字
        input_data: 输入数据
    
    返回:
        包含目标输出的字典，如果失败返回None
    """
    from collections import deque
    print(f"正在查找算子输出: {target_name}")
    print(f"模型: {os.path.basename(model_path)}")
    print("=" * 60)
    try:
        # 加载模型
        model = onnx.load(model_path)
        
        # 找到目标节点
        target_node = None
        for node in model.graph.node:
            if target_name in node.output:
                target_node = node
                break
        
        if target_node is None:
            print(f"错误: 未找到包含输出 {target_name} 的节点")
            return None
        
        # 找到所有依赖节点
        needed_node_names = set()
        needed_inputs = set()
        queue = deque([target_node.name])  # 使用节点名称而不是节点对象
        
        # 创建节点名称到节点的映射
        node_dict = {node.name: node for node in model.graph.node}
        
        while queue:
            current_node_name = queue.popleft()
            if current_node_name in needed_node_names:
                continue
                
            needed_node_names.add(current_node_name)
            current_node = node_dict[current_node_name]
            
            # 添加当前节点的输入
            for input_name in current_node.input:
                needed_inputs.add(input_name)
                
                # 查找产生这个输入的节点
                for node in model.graph.node:
                    if input_name in node.output:
                        queue.append(node.name)  # 使用节点名称
                        break
        
        print(f"找到 {len(needed_node_names)} 个相关节点")
        print(f"需要 {len(needed_inputs)} 个输入")
        
        # 创建新的简化模型
        new_model = onnx.ModelProto()
        new_model.CopyFrom(model)
        
        # 清空所有节点和输出
        new_model.graph.node.clear()
        new_model.graph.output.clear()
        new_model.graph.input.clear()
        new_model.graph.initializer.clear()
        
        # 添加需要的节点
        for node_name in needed_node_names:
            if node_name in node_dict:
                new_model.graph.node.append(node_dict[node_name])
        
        # 添加需要的常量
        for init in model.graph.initializer:
            if init.name in needed_inputs:
                new_model.graph.initializer.append(init)
        
        # 添加需要的输入
        for input_name in needed_inputs:
            # 检查是否已经是常量
            is_constant = False
            for init in new_model.graph.initializer:
                if init.name == input_name:
                    is_constant = True
                    break
            
            if not is_constant:
                input_info = helper.make_tensor_value_info(
                    input_name,
                    onnx.TensorProto.FLOAT16,
                    [1, 1, 1, 1]  # 动态形状
                )
                new_model.graph.input.append(input_info)
        from onnx import helper
        print("=== 模型中的反量化节点 ===")
        for node in new_model.graph.node:
            if node.op_type == "DequantizeLinear":
                print(f"节点名称: {node.name}")
                print(f"  操作类型: {node.op_type}")
                print(f"  输入: {node.input}")
                print(f"  输出: {node.output}")
                
                # 查找相关的初始值
                for init in new_model.graph.initializer:
                    if init.name in node.input:
                        print(f"  初始值 {init.name}:")
                        print(f"    形状: {init.dims}")
                        print(f"    数据类型: {init.data_type}")
                        if init.name.endswith("_scale"):
                            scale_data =get_initializer_data(init)
                            print(f"    Scale值: {scale_data[117]}")
                        elif init.name.endswith("_zero_point"):
                            zp_data = get_initializer_data(init)
                            print(f"    Zero Point值: {zp_data[117]}")
                    
        # 创建tensor value info
        output_info = helper.make_tensor_value_info(
            target_name,
            onnx.TensorProto.FLOAT16,  # 例如: onnx.TensorProto.FLOAT
            input_shape       # 例如: [1, 3, 640, 640]
        )
        
        new_model.graph.output.append(output_info)

        # 保存临时模型
        temp_model_path = "temp_model.onnx"
        onnx.save(new_model, temp_model_path)
        print(f"临时模型已保存到: {temp_model_path}")
        
        # 运行推理
        print("运行推理...")
            # 设置日志级别
        import logging
        logging.basicConfig(level=logging.DEBUG)
        
        # 创建会话时启用详细日志
        session_options = ort.SessionOptions()
        # session_options.log_severity_level = 0  # 0=Verbose, 1=Info, 2=Warning, 3=Error
        # session_options.log_verbosity_level = 0
        session = ort.InferenceSession(temp_model_path, session_options)
 
        outputs = session.run(None, {})

        if not outputs:
            print("推理失败，未获取到输出")
            return None
        
        output_data = outputs[0]
        print(f"输出值: {output_data[(117, 41, 1, 0)]}")
        print(f"\n推理成功！")
        print(f"输出形状: {output_data.shape}")
        print(f"输出类型: {output_data.dtype}")
        print(f"数值范围: [{output_data.min():.6f}, {output_data.max():.6f}]")
        print(f"均值: {output_data.mean():.6f}")
        print(f"标准差: {output_data.std():.6f}")
        
        # 量化特定分析
        if output_data.dtype == np.uint8:
            print(f"量化类型: 8位无符号整数")
            print(f"实际值范围: [{output_data.min()}, {output_data.max()}]")
            print(f"唯一值数量: {len(np.unique(output_data))}")
        elif output_data.dtype == np.int8:
            print(f"量化类型: 8位有符号整数")
            print(f"实际值范围: [{output_data.min()}, {output_data.max()}]")
            print(f"唯一值数量: {len(np.unique(output_data))}")
        elif output_data.dtype == np.float16:
            print(f"量化类型: 16位浮点数")
        elif output_data.dtype == np.float32:
            print(f"数据类型: 32位浮点数")
        
        # 清理临时文件
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)
        
        # 返回结果
        result = {
            'name': target_name,
            'data': output_data,
            'shape': output_data.shape,
            'dtype': output_data.dtype,
            'node_info': {
                'name': target_node.name,
                'op_type': target_node.op_type,
                'inputs': target_node.input,
                'outputs': target_node.output
            },
            'stats': {
                'min': output_data.min(),
                'max': output_data.max(),
                'mean': output_data.mean(),
                'std': output_data.std(),
                'unique_count': len(np.unique(output_data)) if output_data.dtype in [np.uint8, np.int8] else None
            }
        }
        return result
        
    except Exception as e:
        print(f"获取算子输出时出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None

def int32_data_to_float16(int32_data):
    """
    手动将int32_data转换为float16（更精确的方法）
    
    参数:
        int32_data: int32数据列表或numpy数组
    
    返回:
        float16_data: float16数据
    """

    int32_array = np.array(int32_data, dtype=np.int32)

    
    # 方法2：手动转换（更安全）
    float16_list = []
    for int_val in int32_array:
        # 将int32转换为16位二进制字符串
        binary_str = bin(int_val & 0xFFFF)[2:].zfill(16)
        # 将16位二进制转换为uint16
        uint16_val = int(binary_str, 2)
        # 修复：使用正确的字节长度
        float16_val = np.frombuffer(uint16_val.to_bytes(2, byteorder='little'), dtype=np.float16)[0]
        float16_list.append(float16_val)
    return np.array( float16_list)

def create_model_with_specific_output(model_path, target_layer_name, model=None):
    """创建包含特定层输出的模型。传入 model 时可避免多线程并发加载同一文件。"""
    if model is None:
        try:
            model = onnx.load(model_path, load_external_data=True)
        except Exception:
            model = onnx.load(model_path)
    
    # 找到目标层
    target_node = None
    for node in model.graph.node:
        # print(f"node.name: {node.name}")
        if target_layer_name == node.name:
            target_node = node
            break
    
    if target_node is None:
        print(f"警告: 未找到目标层 {target_layer_name}")
        return None
        
    print(f"找到目标层: {target_node.name}")
    print(f"层类型: {target_node.op_type}")
    print(f"输入: {target_node.input}")
    print(f"输出: {target_node.output}")
    # 获取层的输入输出信息
    value_info = {}
    for info in model.graph.value_info:
        value_info[info.name] = info
    
    # 创建新的输出信息
    new_outputs = []
    output_name = []
    # 添加目标层的输出
    
    if target_node.output[0] in value_info:
        output_info = value_info[target_node.output[0]]
        new_outputs.append(output_info)
        output_name.append(output_info.name)
        print(f"添加输出: {target_node.output[0]}")
    else:
        print(f"警告: 未在value_info中找到 {target_node.output[0]}")
           
        

    # 添加目标层的输入
    image = None
    bias_name = None
    weight_name = None
    for input_name in target_node.input:
        if input_name in value_info:
            input_info = value_info[input_name]
            new_outputs.append(input_info)
            output_name.append(input_info.name)
            print(f"添加输入: {input_name}")
        else:
            if 'image' in input_name.lower():
                output_name.append('image')
        if 'bias' in input_name.lower():
            bias_name = input_name
        if 'weight' in input_name.lower():
            weight_name = input_name
        else:
            print(f"警告: 未在value_info中找到输入 {input_name}")
    # 清空原有输出，添加新的输出
    # model.graph.output.clear()
    # model.graph.output.extend(new_outputs)
    
    # 保存修改后的模型
    # onnx.save(model, output_path)
    bias_data = None
    scale_data = None
    weight_quantized_data = None
    input_data = None
    weight_data = None
    for initializer in model.graph.initializer:
        if (bias_name is not None and initializer.name == bias_name) or (weight_name is not None and weight_name.split('_')[0] in initializer.name):
            # 使用 onnx.numpy_helper.to_array 统一解析（支持 raw_data / float_data / int32_data / 已加载的外部数据）
            try:
                data = onnx_tensor_to_array(initializer)
            except Exception as e:
                print(f"警告: 无法读取 initializer {initializer.name}: {e}")
                continue
            if data.size == 0:
                print(f"警告: initializer {initializer.name} 数据为空，跳过")
                continue

            if initializer.name == bias_name:
                bias_data = data
            else:
                if 'scale' in initializer.name:
                    scale_data = data
                elif 'weight_quantized' in initializer.name:
                    weight_quantized_data = data
    conv_info = {}
    if target_node is not None and target_node.op_type == 'Conv':
        for attr in target_node.attribute:
            if attr.name == 'kernel_shape':
                conv_info['kernel_shape'] = attr.ints
            elif attr.name == 'strides':
                conv_info['strides'] = attr.ints
            elif attr.name == 'pads':
                conv_info['pads'] = attr.ints
            
    return output_name, bias_data, scale_data, weight_quantized_data, conv_info

_PROBE_IMAGE_NAME = (
    "2025_03_01_08_24_28_2826_-0.02635467520423127_0.2131570949018095_"
    "-0.975110744245263_-0.055031994742602286_290035.98_-31045.848_-15416.997_earth_.png"
)
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def _first_image(folder):
    if not os.path.isdir(folder):
        return None
    names = sorted(
        f
        for f in os.listdir(folder)
        if f.lower().endswith(_IMAGE_EXTS) and os.path.isfile(os.path.join(folder, f))
    )
    return os.path.join(folder, names[0]) if names else None


def resolve_generate_bin_image(model_path):
    """抽 bin 探针图：环境变量 > 引擎旁遗留 cali_data > 旧项目 cali_data > 本任务 input/cali。"""
    env = os.environ.get("GENERATE_BIN_IMAGE", "").strip()
    if env and os.path.isfile(env):
        return os.path.abspath(env)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_probe = os.path.join(script_dir, "cali_data", _PROBE_IMAGE_NAME)
    if os.path.isfile(local_probe):
        return local_probe
    platform_root = os.path.dirname(os.path.dirname(script_dir))
    legacy = os.environ.get("QUANTITIZE_LEGACY_DIR", "").strip() or os.path.join(
        os.path.dirname(platform_root), "quantitize"
    )
    legacy_probe = os.path.join(legacy, "cali_data", _PROBE_IMAGE_NAME)
    if os.path.isfile(legacy_probe):
        return os.path.abspath(legacy_probe)
    job_cali = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(model_path))), "input", "cali")
    img = _first_image(job_cali)
    if img:
        return img
    # scratch 布局下 model 位于 TASK_SCRATCH_ROOT/<task_id>/workspace，
    # 而持久输入仍在 OUTPUT_DATA_ROOT/<task_id>/input。
    output_root = os.environ.get("OUTPUT_DATA_ROOT", "").strip()
    scratch_root = os.environ.get("TASK_SCRATCH_ROOT", "").strip()
    if output_root and scratch_root:
        try:
            task_id = os.path.relpath(os.path.abspath(model_path), os.path.abspath(scratch_root)).split(os.sep)[0]
            persistent_cali = os.path.join(output_root, task_id, "input", "cali")
            img = _first_image(persistent_cali)
            if img:
                return img
        except (OSError, ValueError):
            pass
    img = _first_image(os.path.join(legacy, "cali_data"))
    if img:
        return img
    raise FileNotFoundError(
        "未找到 generate_bin 探针图。可设置 GENERATE_BIN_IMAGE，"
        "或保证任务 input/cali 有图，或保留旧项目 cali_data。"
    )


def run_inference(quantized_model_path):
    image_path = resolve_generate_bin_image(quantized_model_path)
    print("准备输入数据...")
    image = load_and_preprocess_real_image(image_path)
    input_data = {'images': image}
    
    quantized_session = ort.InferenceSession(quantized_model_path)
    quantized_results = quantized_session.run(None, input_data)
    return quantized_results, [i.name for i in  quantized_session.get_outputs()],input_data

def export_specific_layer_io(quantized_results, input_data, names, model_path, target_layer_name, model=None):
    """
    导出特定层的输入输出数据
    
    参数:
        quantized_model_path: 量化模型路径  
        input_data: 输入数据
        target_layer_name: 目标层名称 (例如: '/model.4/m.5/cv1/conv/Conv_output_0')
        output_folder: 输出文件夹
    """
    print(f"=== 导出特定层数据: {target_layer_name} ===")
    
    
    print("\n--- 处理量化模型 ---")
    target_output_name, bias_data, scale_data, weight_quantized_data, conv_info = create_model_with_specific_output(model_path, target_layer_name, model=model)
    
    # 2. 运行推理并获取特定层的数据
    print("\n--- 运行推理获取特定层数据 ---")
    
    try:
        # 加载修改后的模型

        
        # 保存原始模型的数据
        result = {}
        print("\n=== 原始模型数据 ===")
        for i, output in enumerate(quantized_results):
            output_name = names[i]
            if output_name in target_output_name:
                data = quantized_results[i]
                if output_name == target_output_name[0]:
                    result["output"] = data
                else:
                    result['input'] = data
        
        if 'input' not in result or 'output' not in result:
            if 'image' in target_output_name:
                result['input'] = input_data['images']
        # if scale_data is None:
        #     import ipdb; ipdb.set_trace()
        result['scale'] = scale_data
        result['weight_quantized'] = weight_quantized_data
        # bias 未找到或加载失败时用与 scale 同形状的零向量，避免后续 column_stack 报错
        if bias_data is not None:
            result['bias'] = bias_data
        elif scale_data is not None:
            result['bias'] = np.zeros(np.asarray(scale_data).shape, dtype=np.float32)
        else:
            result['bias'] = bias_data
        result['conv_info'] = conv_info
    except Exception as e:
        print(f"导出过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    return result

def extract_conv_weights(model_path, target_layer_name):
    """从模型中提取卷积层的权重"""
    model = onnx.load(model_path)
    
    # 找到目标层
    target_node = None
    for node in model.graph.node:
        if node.output and target_layer_name in node.output:
            target_node = node
            break
    
    if target_node is None:
        print(f"警告: 未找到目标层 {target_layer_name}")
        return None, None, None
        
    return  target_node

def debug_channel(orig_weight, quant_weight, ch):
    """
    专门调试通道117的量化问题 - 对比uint8对称量化和非对称量化
    """
    orig_ch = orig_weight[ch]
    quant_ch = quant_weight[ch]
    
    print(f"=== 通道 {ch} 详细调试 ===")
    
    # 1. 检查权重分布
    print("1. 权重分布分析:")
    print(f"  原始权重范围: [{orig_ch.min():.8f}, {orig_ch.max():.8f}]")
    print(f"  原始权重均值: {orig_ch.mean():.8f}")
    print(f"  原始权重标准差: {orig_ch.std():.8f}")
    
    # 2. Uint8对称量化
    print("2. Uint8对称量化:")
    abs_max = np.max(np.abs(orig_ch))
    scale_sym = np.float16(abs_max / 127.0)  # 对称量化，范围是[-127, 127]
    
    manual_quant_sym = np.round(orig_ch / scale_sym)
    manual_quant_sym = np.clip(manual_quant_sym, -127, 127)
    # 转换为uint8: 将[-127, 127]映射到[0, 254]
    manual_quant_sym_uint8 = (manual_quant_sym + 127).astype(np.uint8)
    manual_dequant_sym = (manual_quant_sym_uint8.astype(np.float16) - 127) * scale_sym
    diff_sym = orig_ch - manual_dequant_sym
    
    print(f"  对称量化scale: {scale_sym:.8f}")
    print(f"  对称量化范围: [{manual_quant_sym_uint8.min()}, {manual_quant_sym_uint8.max()}]")
    print(f"  对称量化误差: MAE={np.mean(np.abs(diff_sym)):.8f}, Max={np.max(np.abs(diff_sym)):.8f}")
    
    # 3. Uint8非对称量化
    print("3. Uint8非对称量化:")
    # 计算uint8量化参数 (非对称量化)
    rmin = orig_ch.min()
    rmax = orig_ch.max()
    scale_asym = np.float16((rmax - rmin) / 255.0)
    zero_point_asym = np.round(-rmin / scale_asym)
    zero_point_asym = np.clip(zero_point_asym, 0, 255)
    
    manual_quant_asym = np.round(orig_ch / scale_asym + zero_point_asym)
    manual_quant_asym = np.clip(manual_quant_asym, 0, 255)
    manual_quant_asym_uint8 = manual_quant_asym.astype(np.uint8)
    manual_dequant_asym = (manual_quant_asym_uint8.astype(np.float16) - zero_point_asym) * scale_asym
    diff_asym = orig_ch - manual_dequant_asym
    
    print(f"  非对称量化scale: {scale_asym:.8f}")
    print(f"  非对称量化zero_point: {zero_point_asym}")
    print(f"  非对称量化范围: [{manual_quant_asym_uint8.min()}, {manual_quant_asym_uint8.max()}]")
    print(f"  非对称量化误差: MAE={np.mean(np.abs(diff_asym)):.8f}, Max={np.max(np.abs(diff_asym)):.8f}")
    
    # 4. 对比分析
    print("4. 对称 vs 非对称 对比:")
    mae_ratio = np.mean(np.abs(diff_asym)) / np.mean(np.abs(diff_sym))
    max_ratio = np.max(np.abs(diff_asym)) / np.max(np.abs(diff_sym))
    
    print(f"  MAE误差比 (非对称/对称): {mae_ratio:.2f}x")
    print(f"  最大误差比 (非对称/对称): {max_ratio:.2f}x")
    print(f"  Scale精度差异: {abs(scale_sym - scale_asym):.2e}")
    
    # 5. 量化范围利用率对比
    print("5. 量化范围利用率对比:")
    sym_range_usage = (manual_quant_sym_uint8.max() - manual_quant_sym_uint8.min() + 1) / 256 * 100
    asym_range_usage = (manual_quant_asym_uint8.max() - manual_quant_asym_uint8.min() + 1) / 256 * 100
    
    print(f"  对称量化使用率: {sym_range_usage:.1f}%")
    print(f"  非对称量化使用率: {asym_range_usage:.1f}%")
    print(f"  使用率差异: {asym_range_usage - sym_range_usage:.1f}%")
    
    # 6. 权重分布对称性分析
    print("6. 权重分布对称性分析:")
    mean_val = orig_ch.mean()
    abs_mean = np.abs(mean_val)
    range_val = orig_ch.max() - orig_ch.min()
    symmetry_ratio = abs_mean / range_val if range_val > 0 else 0
    
    print(f"  权重均值: {mean_val:.8f}")
    print(f"  权重范围: {range_val:.8f}")
    print(f"  对称性比例: {symmetry_ratio:.6f}")
    if symmetry_ratio < 0.1:
        print("  建议: 分布接近对称，适合对称量化")
    else:
        print("  建议: 分布不对称，适合非对称量化")
    
    # 7. 找出误差最大的位置
    print("7. 最大误差位置对比:")
    max_error_idx_sym = np.unravel_index(np.argmax(np.abs(diff_sym)), diff_sym.shape)
    max_error_idx_asym = np.unravel_index(np.argmax(np.abs(diff_asym)), diff_sym.shape)
    
    print(f"  对称量化最大误差位置: {max_error_idx_sym}")
    print(f"    原始值: {orig_ch[max_error_idx_sym]:.8f}")
    print(f"    量化值: {manual_quant_sym_uint8[max_error_idx_sym]:.8f}")
    print(f"    反量化值: {manual_dequant_sym[max_error_idx_sym]:.8f}")
    print(f"    误差: {diff_sym[max_error_idx_sym]:.8f}")
    
    print(f"  非对称量化最大误差位置: {max_error_idx_asym}")
    print(f"    原始值: {orig_ch[max_error_idx_asym]:.8f}")
    print(f"    量化值: {manual_quant_asym_uint8[max_error_idx_asym]:.8f}")
    print(f"    反量化值: {manual_dequant_asym[max_error_idx_asym]:.8f}")
    print(f"    误差: {diff_asym[max_error_idx_asym]:.8f}")
    
    # 8. 量化精度对比
    print("8. 量化精度对比:")
    print(f"  对称量化精度: {scale_sym:.8f}")
    print(f"  非对称量化精度: {scale_asym:.8f}")
    print(f"  精度比: {scale_sym / scale_asym:.2f}x")
    
    return {
        'scale_sym': scale_sym,
        'scale_asym': scale_asym,
        'zero_point_asym': zero_point_asym,
        'mae_sym': np.mean(np.abs(diff_sym)),
        'mae_asym': np.mean(np.abs(diff_asym)),
        'max_error_sym': np.max(np.abs(diff_sym)),
        'max_error_asym': np.max(np.abs(diff_asym)),
        'mae_ratio': mae_ratio,
        'max_ratio': max_ratio,
        'symmetry_ratio': symmetry_ratio
    }
def analyze_weight_quantization(quantized_model_path, target_layer_name, output_folder, input_data=None):
    """
    分析权重量化的影响
    """
    print(f"=== 分析权重量化: {target_layer_name} ===")
    
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    
    # 提取量化模型权重
    print("\n--- 提取量化模型权重 ---")
    quant_weight, quant_bias = extract_conv_weights(quantized_model_path, target_layer_name,input_data=input_data, input_shape=orig_weight.shape)
    
    return quant_weight, quant_bias



def get_quant_weight(results, input_data, names, model_path, target_layer, model=None):
    """主函数。传入 model 时可避免多线程并发加载同一文件。"""
    quantized_results = export_specific_layer_io(results, input_data, names, model_path, target_layer, model=model)
    return quantized_results

# 在导入 creat_bin 之前清理 sys.path，移除 Isaac Sim 等可能冲突的路径
import sys
sys.path = [p for p in sys.path if p and 'isaac-sim' not in p.lower()]

from creat_bin import creat_bin
def save_data_as_text(data, file_path, dtype='float16', is_txt=False, is_npy=False):
    """将numpy数组保存为文本格式"""
    # 确保目录存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # 保存为文本格式
    if dtype == 'float16':
        fmt = '%.6f'
        data = data.astype(np.float16)
    else:
        fmt = '%d'
        data = data.astype(np.short)
    if is_npy:
        print(file_path)
        np.save(file_path, data)
    if is_txt:
        np.savetxt(file_path, data.flatten(), fmt=fmt, delimiter=' ')
    else:
        creat_bin(data,  file_path)
    # np.savetxt(file_path, data.flatten(), fmt=fmt, delimiter=' ')
    

def save_cbs(results, input_data, names, model_path, folder_name, layer_name, cnt, folder_path, is_txt=False, model=None):
    out_dir = os.path.join(folder_path.rstrip(os.sep), folder_name)
    os.makedirs(out_dir, exist_ok=True)
    result = get_quant_weight(results, input_data, names, model_path, f'{layer_name}/conv/Conv', model=model)
    fp_input   = result['input']
    int_wt     = result['weight_quantized']
    fp_bn      =  np.column_stack((result['bias'], result['scale'] * 1000)).astype(np.float16)
    fp_golden  = result['output']

    # 保存为文本格式
    save_data_as_text(fp_input, folder_path + folder_name + "/" + folder_name + "_conv_input.txt", is_txt=is_txt)
    save_data_as_text(int_wt, folder_path + folder_name + "/" + folder_name + "_conv_wt.txt", dtype='int8', is_txt=is_txt)
    save_data_as_text(fp_bn, folder_path + folder_name + "/" + folder_name + "_conv_bn.txt", is_txt=is_txt)
    save_data_as_text(fp_golden, folder_path + folder_name + "/" + folder_name + "_conv_output.txt", is_txt=is_txt)
    result = get_quant_weight(results, input_data, names, model_path, f'{layer_name}/act/Mul', model=model)
    fp_golden  = result['output']
    save_data_as_text(fp_golden, folder_path + folder_name + "/" + folder_name + "_silu_output.txt", is_txt=is_txt)
    cnt += 1
    # import ipdb; ipdb.set_trace()
    return cnt

def save_conv(results, input_data, names, model_path, folder_name, layer_name, cnt, folder_path, is_txt=False, model=None):
    out_dir = os.path.join(folder_path.rstrip(os.sep), folder_name)
    os.makedirs(out_dir, exist_ok=True)
    result = get_quant_weight(results, input_data, names, model_path, f'{layer_name}/Conv', model=model)
    fp_input   = result['input']
    int_wt     = result['weight_quantized']
    fp_bn      =  np.column_stack((result['bias'], result['scale'] * 1000)).astype(np.float16)
    fp_golden  = result['output']
    save_data_as_text(fp_input, folder_path + folder_name + "/" + folder_name + "_conv_input.txt", is_txt=is_txt)
    save_data_as_text(int_wt, folder_path + folder_name + "/" + folder_name + "_conv_wt.txt", dtype='int8', is_txt=is_txt)
    save_data_as_text(fp_bn, folder_path + folder_name + "/" + folder_name + "_conv_bn.txt", is_txt=is_txt)
    save_data_as_text(fp_golden, folder_path + folder_name + "/" + folder_name + "_conv_output.txt", is_txt=is_txt)
    return cnt

def save_add(results, input_data, names, model_path, layer_idx, operator, cnt, layer_name, folder_path, is_txt=False, model=None):
    folder_name = f'T32_L{layer_idx:02d}_s{cnt:02d}'
    if 'c2f' in layer_name:
        result = get_quant_weight(results, input_data, names, model_path, operator, model=model)
        fp_golden  = result['output']
        save_data_as_text(fp_golden, folder_path + folder_name + "/" + folder_name + "_shortcut_output.txt", is_txt=is_txt)
    return cnt

def save_extra_layer(results, input_data, names, model_path, layer_idx, operator, cnt, folder_path, name, is_txt=False, model=None):
    folder_name = f'T32_L{layer_idx:02d}_s{cnt:02d}'
    result = get_quant_weight(results, input_data, names, model_path, operator, model=model)
    fp_golden  = result['output']
    fp_input = result['input']
    save_data_as_text(fp_golden, folder_path + folder_name + "/" + folder_name + f"_{name}_output.txt", is_txt=is_txt)
    save_data_as_text(fp_input, folder_path + folder_name + "/" + folder_name + f"_{name}_input.txt", is_txt=is_txt)
    cnt += 1
    return cnt

def save_head(results, input_data, names, model_path, layer_idx, folder_path, model=None):
    output_list = [
        # ('/model.30/Sub','output','tmp_x1y1_output'),
        # ('/model.30/Add_1','output','tmp_x2y2_output'),
        # ('/model.30/Concat_6','output','tmp_output'),
        # ('/model.30/Add_2','output','tmp_x1y1+x2y2_output'),
        # ('/model.30/Sub_1','output','tmp_x1yq-x2y2_output'),

        # ('/model.30/Concat_6','output','tmp_output'),
        #start
        # NOTE(20260727): Pose P6 导出图最终输出为 Concat_5→output0，无 Concat_8。
        # 原硬编码 Concat_8 会导致 generate_bin L30 失败。恢复方法见本任务 debug/debug.md。
        ('/model.30/Concat_5','output','final_output'),
        ('/model.30/dfl/Transpose','output','transpose_output'),
        ('/model.30/dfl/Softmax','output','softmax_output'),
        # Concat_5 已作为 final_output，不再重复导出 concat_5_output
        # ('/model.30/Concat_5', 'output', 'concat_5_output'),
        ('/model.30/dfl/Reshape','input','dfl_intput'),
        # ('/model.30/dfl/Reshape_1','output','z`'),
        ('/model.30/dfl/Reshape_1','output','Dist2box_input'),
        ('/model.30/Mul_2','output','Dist2box_output'),
        ('/model.30/Sigmoid','output','sigmoid_output'),
        ('/model.30/Mul_3','output','mul_3_output'),
        ('/model.30/Concat', 'output', 'concat_output'),
        ('/model.30/Reshape_9','output','kpts_decode_output'),
        #end
        ('/model.30/Sigmoid','input','Sigmoid_input_1'),
    ]
    output_layers =[
        "/model.30/cv2.0/cv2.0.0",
        "/model.30/cv2.1/cv2.1.0",
        "/model.30/cv2.2/cv2.2.0",
        "/model.30/cv2.3/cv2.3.0",
        "/model.30/cv3.0/cv3.0.0",          
        "/model.30/cv3.1/cv3.1.0",
        "/model.30/cv3.2/cv3.2.0",
        "/model.30/cv3.3/cv3.3.0",
        "/model.30/cv4.0/cv4.0.0",
        "/model.30/cv4.1/cv4.1.0",
        "/model.30/cv4.2/cv4.2.0",
        "/model.30/cv4.3/cv4.3.0"
    ]
    for i, output_layer in enumerate(output_layers):
        layer_idx = i + 31
        cnt = 0
        save_cbs(results, input_data, names, model_path, f'T32_L{layer_idx:02d}_s{cnt:02d}', output_layer, cnt, folder_path, is_txt=False, model=model)
        print(output_layer)
        cnt += 1
        save_cbs(results, input_data, names, model_path, f'T32_L{layer_idx:02d}_s{cnt:02d}', output_layer[:-1] + '1', cnt, folder_path, is_txt=False, model=model)
        print(output_layer[:-1] + '1')
        cnt += 1
        save_conv(results, input_data, names, model_path, f'T32_L{layer_idx:02d}_s{cnt:02d}', output_layer[:-1] + '2', cnt, folder_path, is_txt=False, model=model)
        print(output_layer[:-1] + '2')

    if not os.path.exists(folder_path + 'extra_layer'):
        os.makedirs(folder_path + 'extra_layer', exist_ok=True)
    for layer, key, output_name in output_list:
        result = get_quant_weight(results, input_data, names, model_path, layer, model=model)
        try:
            if 'tmp' in output_name:
                save_data_as_text(result[key], folder_path + 'extra_layer' + "/" +  f"{output_name}.npy", is_npy=False)
            else:
                save_data_as_text(result[key], folder_path + 'extra_layer' + "/" +  f"{output_name}.txt", is_txt=True)
        except:
            # import ipdb; ipdb.set_trace()
            raise
    

def save_layer(results, input_data, names, model_path, layer_idx, cnt, operators, folder_path, save_txt=False, model=None):
    i = 0
    pass_list = []
    while(i < len(operators)):
        operator = operators[i]
        if operator in pass_list:
            i = i + 1
            continue
        if 'conv' in operator:
            t_name = operator.replace('/conv/Conv', '')
            if t_name + '/act/Mul' in operators:
                print('save_cbs ', t_name)
                pass_list.append(t_name + '/act/Mul')
                cnt = save_cbs(results, input_data, names, model_path, f'T32_L{layer_idx:02d}_s{cnt:02d}', operator.replace('/conv/Conv', ''), cnt, folder_path, is_txt=save_txt, model=model)

        elif 'Add' in operator:
            print('Add')
            save_add(results, input_data, names, model_path, layer_idx, operator, cnt - 1, 'c2f', folder_path, is_txt=save_txt, model=model)
        elif 'MaxPool' in operator:
            print('save_maxpool')
            cnt = save_extra_layer(results, input_data, names, model_path, layer_idx, operator, cnt, folder_path, 'maxpool', is_txt=save_txt, model=model)
        elif 'Resize' in operator:
            print('Resize')
            cnt = save_extra_layer(results, input_data, names, model_path, layer_idx, operator, cnt, folder_path, 'upsample', is_txt=save_txt, model=model)
        elif 'Concat' in operator:
            print('concat')
            cnt = save_extra_layer(results, input_data, names, model_path, layer_idx, operator, cnt, folder_path, 'shortcut', is_txt=save_txt, model=model)
        elif 'Conv' in operator:
            print('save_conv', t_name)
        else:
            print(f'unknown layer {operator}')
        i = i + 1
    return cnt

# def save_SPPF(results, input_data,names, model_path, layer_idx,cnt, operators ,folder_path):
#     for operator in operators:
#         if 'conv' in operator:
#             cnt = save_cbs(results, input_data,names, model_path, f'T32_L{layer_idx:02d}_s{cnt:02d}', operator.replace('/conv/Conv', ''), cnt, folder_path)
#     return cnt

# def save_c2(results, input_data,names, model_path, layer_idx, cnt, operators,folder_path):
#     for operator in operators:
#         if 'conv' in operator:
#             cnt = save_cbs(results, input_data,names, model_path, f'T32_L{layer_idx:02d}_s{cnt:02d}', operator.replace('/conv/Conv', ''), cnt, folder_path)
    return cnt
 
def _process_one_layer(layer_idx, results, input_data, names, model_path, operators, folder_path, model=None):
    """处理单层，供多线程调用。传入 model 时复用主线程已加载的模型，避免并发读同一文件。"""
    if layer_idx != 30:
        operators1 = [i for i in operators if f'model.{layer_idx}/' in i]
        save_layer(results, input_data, names, model_path, layer_idx, 0, operators1, folder_path, save_txt=False, model=model)
    else:
        save_head(results, input_data, names, model_path, layer_idx, folder_path, model=model)


NUM_LAYERS = 32  # 导出层 0～31

if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from get_conv_name import load_onnx_operators
    # model_path = "/home/crab2/yolov8/Quantization-YOLOv8-main/myquant2/temp_model2/"
    # model_path = "/home/crab2/yolov8/Quantization-YOLOv8-main/myquant2/temp_model2/wrs_fp16_final2_output.onnx"
    model_path = sys.argv[1]
    folder_path = sys.argv[2]
    if folder_path[-1] != '/':
        folder_path = folder_path + '/'
    results, names, intput_data = run_inference(model_path)
    operators = load_onnx_operators(model_path)

    # 主线程只加载一次模型，传入各 worker 复用，避免多线程并发读同一文件导致 initializer 数据不完整
    try:
        shared_model = onnx.load(model_path, load_external_data=True)
    except Exception:
        shared_model = onnx.load(model_path)

    max_workers = min(8, NUM_LAYERS)
    failed_layers = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_one_layer,
                layer_idx, results, intput_data, names, model_path, operators, folder_path, shared_model,
            ): layer_idx
            for layer_idx in range(NUM_LAYERS)
        }
        for fut in as_completed(futures):
            layer_idx = futures[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"layer_idx={layer_idx} 出错: {e}", file=sys.stderr)
                failed_layers.append((layer_idx, str(e)))
    if failed_layers:
        detail = "; ".join(f"L{idx}: {err}" for idx, err in sorted(failed_layers))
        print(f"generate_bin 失败: {len(failed_layers)} 层出错 — {detail}", file=sys.stderr)
        sys.exit(1)
