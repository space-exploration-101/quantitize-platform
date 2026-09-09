import os
import re
import sys
import numpy as np
import shutil

# 尝试导入正确的 shapes
try:
    # 添加 bin_process 目录到路径
    bin_process_dir = os.path.join(os.path.dirname(__file__), '..', 'bin_process')
    if bin_process_dir not in sys.path:
        sys.path.insert(0, bin_process_dir)
    from layer_wt_shapes import WT_SHAPES
    HAS_WT_SHAPES = True
except ImportError:
    WT_SHAPES = None
    HAS_WT_SHAPES = False
    print("警告: 无法导入 layer_wt_shapes.WT_SHAPES，将无法进行大小调整")

# 导入 legacy 特殊层处理脚本
try:
    import L02_s00_wt_bin_process as l02_s00_wt_proc
    import L02_s07_wt_bin_process as l02_s07_wt_proc
    HAS_LEGACY_SPECIAL_PROCESSORS = True
except ImportError:
    HAS_LEGACY_SPECIAL_PROCESSORS = False
    print("警告: 无法导入 L02_s00/L02_s07 legacy 处理脚本，将使用默认的补零方式")


def process_l02_s00_wt_via_legacy_script(weight_data):
    """直接复用 L02_s00_wt_bin_process.py 的 demap/parallel 处理流程。"""
    l02_s00_wt_proc.Tout = 32
    l02_s00_wt_proc.Tin = 32
    int_weight = l02_s00_wt_proc.demap_of_weight(weight_data, 160, 160, 1, 1)

    new_int_weight = np.zeros((192, 160, 1, 1), dtype=np.int8)
    for i in range(192):
        for j in range(160):
            if i < 80:
                new_int_weight[i, j, 0, 0] = int_weight[i, j, 0, 0]
            elif i < 96:
                new_int_weight[i, j, 0, 0] = 0
            elif i < 176:
                new_int_weight[i, j, 0, 0] = int_weight[i - 16, j, 0, 0]
            else:
                new_int_weight[i, j, 0, 0] = 0

    t_new_int_weight = l02_s00_wt_proc.parallel_weight(new_int_weight, 192, 160, 1, 1)
    return t_new_int_weight.flatten().astype(np.int8)


def process_l02_s07_wt_via_legacy_script(weight_data):
    """直接复用 L02_s07_wt_bin_process.py 的 demap/parallel 处理流程。"""
    l02_s07_wt_proc.Tout = 32
    l02_s07_wt_proc.Tin = 32
    int_weight = l02_s07_wt_proc.demap_of_weight(weight_data, 160, 416, 1, 1)

    new_int_weight = np.zeros((160, 480, 1, 1), dtype=np.int8)
    for i in range(160):
        for j in range(480):
            if j < 80:
                new_int_weight[i, j, 0, 0] = int_weight[i, j, 0, 0]
            elif j < 96:
                new_int_weight[i, j, 0, 0] = 0
            elif j < 176:
                new_int_weight[i, j, 0, 0] = int_weight[i, j - 16, 0, 0]
            elif j < 192:
                new_int_weight[i, j, 0, 0] = 0
            elif j < 272:
                new_int_weight[i, j, 0, 0] = int_weight[i, j - 32, 0, 0]
            elif j < 288:
                new_int_weight[i, j, 0, 0] = 0
            elif j < 368:
                new_int_weight[i, j, 0, 0] = int_weight[i, j - 48, 0, 0]
            elif j < 384:
                new_int_weight[i, j, 0, 0] = 0
            elif j < 464:
                new_int_weight[i, j, 0, 0] = int_weight[i, j - 64, 0, 0]
            else:
                new_int_weight[i, j, 0, 0] = 0

    t_new_int_weight = l02_s07_wt_proc.parallel_weight(new_int_weight, 160, 480, 1, 1)
    return t_new_int_weight.flatten().astype(np.int8)


def extract_layer_info(filename):
    """
    从文件名中提取层信息 L{layer}_{sub} 或 L{layer}
    
    Args:
        filename: 文件名，例如 "T32_T32_L02_s00_conv_wt.bin" 或 "L02_s00_conv_wt.bin" 或 "T32_T32_L43_WT.bin"
    
    Returns:
        (layer, sub) 元组，例如 (2, 0) 或 (43, 0)，如果无法提取则返回 None
    """
    # 匹配 L{数字}_s{数字} 的模式
    match = re.search(r'L(\d+)_s(\d+)', filename)
    if match:
        layer = int(match.group(1))
        sub = int(match.group(2))
        return (layer, sub)
    # 匹配 L{数字} 的模式（如 L43）
    match = re.search(r'L(\d+)(?:_WT|_wt|_conv_wt)?', filename)
    if match:
        layer = int(match.group(1))
        return (layer, 0)  # L43 等层没有 sub，默认为 0
    return None




def adjust_weight_size(weight_data, target_shape):
    """
    调整权重数据的大小以匹配目标shape
    
    Args:
        weight_data: numpy数组，int8类型
        target_shape: 目标shape元组，例如 (192, 160, 1, 1)
    
    Returns:
        调整后的权重数据
    """
    target_size = np.prod(target_shape)
    actual_size = weight_data.size
    
    if actual_size == target_size:
        return weight_data
    elif actual_size < target_size:
        # 需要补零
        padding_size = target_size - actual_size
        padding = np.zeros(padding_size, dtype=np.int8)
        adjusted = np.concatenate([weight_data, padding])
        print(f"  补零: {actual_size} -> {target_size} (+{padding_size})")
        return adjusted
    else:
        # 需要截取
        adjusted = weight_data[:target_size]
        print(f"  截取: {actual_size} -> {target_size} (-{actual_size - target_size})")
        return adjusted


def copy_and_adjust_bin_files(source_folder, output_folder):
    """
    复制bin文件到新文件夹，调整大小以匹配正确的shape，保持原有文件名不变

    Args:
        source_folder: 源文件夹路径
        output_folder: 输出文件夹路径
    """
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 获取所有bin文件
    file_paths = []
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            # 匹配 wt.bin 或 _WT.bin（大小写不敏感）
            if file.endswith('.bin') and ('wt.bin' in file.lower() or '_wt.bin' in file.lower()):
                file_paths.append(os.path.join(root, file))

    if not file_paths:
        print(f"在文件夹 {source_folder} 中未找到bin文件")
        return

    print(f"找到 {len(file_paths)} 个bin文件")

    # 提取层信息并排序
    def extract_sort_key(filepath):
        filename = os.path.basename(filepath)
        layer_info = extract_layer_info(filename)
        if layer_info:
            layer, sub = layer_info
            return (layer, sub)
        # 如果无法提取，尝试从文件名中提取数字
        numbers = re.findall(r'\d+', filename)
        if numbers:
            return (int(numbers[0]) if len(numbers) > 0 else 0, 
                   int(numbers[1]) if len(numbers) > 1 else 0)
        return (0, 0)

    # 按层信息排序
    sorted_files = sorted(file_paths, key=extract_sort_key)
    
    if not HAS_WT_SHAPES:
        print("警告: 无法导入 WT_SHAPES，将不进行大小调整")
    else:
        # 导入辅助函数
        try:
            from layer_wt_shapes import get_wt_shape
        except ImportError:
            get_wt_shape = None
            print("警告: 无法导入 get_wt_shape 函数")

    # 复制、调整大小，保持原有文件名
    for i, source_path in enumerate(sorted_files):
        filename = os.path.basename(source_path)
        # 保持原有文件名，不进行重命名
        new_path = os.path.join(output_folder, filename)
        
        # 读取原始文件
        weight_data = np.fromfile(source_path, dtype=np.int8)
        original_size = weight_data.size
        
        # 如果可以使用 WT_SHAPES，调整大小
        if HAS_WT_SHAPES and get_wt_shape:
            # 从文件名提取层名（例如 "T32_T32_L02_s00_conv_wt.bin" -> "L02_s00"）
            layer_info = extract_layer_info(filename)
            if layer_info:
                layer, sub = layer_info
                layer_name = f"L{layer:02d}_s{sub:02d}"
                target_shape = get_wt_shape(layer_name)
                
                if target_shape:
                    target_size = np.prod(target_shape)
                    
                    if original_size != target_size:
                        print(f"文件 {i}: {filename}")
                        print(f"  层名: {layer_name}")
                        print(f"  原始大小: {original_size}, 目标大小: {target_size} (shape: {target_shape})")
                        
                        # 检查是否是特殊层，需要使用 legacy 脚本处理
                        if HAS_LEGACY_SPECIAL_PROCESSORS:
                            if layer_name == "L02_s00":
                                # L02_s00: 直接调用 L02_s00_wt_bin_process.py 对应逻辑
                                if original_size == 25600:  # 160 * 160 * 1 * 1
                                    print(f"  使用 legacy 脚本处理: L02_s00")
                                    weight_data = process_l02_s00_wt_via_legacy_script(weight_data)
                                elif original_size == 30720:  # 已经是192通道，但可能是简单补零的
                                    print(f"  检测到192输出通道数据，检查是否需要重新处理...")
                                    # 检查后32输出通道是否全为0（简单补零的特征）
                                    # 后32输出通道 = 32 * 160 = 5120 字节
                                    last_32_channels = weight_data[-5120:]
                                    if np.all(last_32_channels == 0):
                                        print(f"  检测到简单补零，还原到160输出通道后重新处理")
                                        # 还原到160输出通道
                                        weight_data_160ch = weight_data[:25600]
                                        # 应用 legacy 脚本处理
                                        weight_data = process_l02_s00_wt_via_legacy_script(weight_data_160ch)
                                    else:
                                        print(f"  数据已包含非零值，保持原样")
                                else:
                                    print(f"  警告: L02_s00 原始大小 {original_size} 不符合预期 (25600或30720)，使用默认补零")
                                    weight_data = adjust_weight_size(weight_data, target_shape)
                            elif layer_name == "L02_s07":
                                # L02_s07: 直接调用 L02_s07_wt_bin_process.py 对应逻辑
                                if original_size == 66560:  # 160 * 416 * 1 * 1
                                    print(f"  使用 legacy 脚本处理: L02_s07")
                                    weight_data = process_l02_s07_wt_via_legacy_script(weight_data)
                                elif original_size == 76800:  # 已经是480输入通道，但可能是简单补零的
                                    print(f"  检测到480输入通道数据，检查是否需要重新处理...")
                                    # 检查后64输入通道是否全为0（简单补零的特征）
                                    # 后64输入通道 = 160 * 64 = 10240 字节
                                    last_64_channels = weight_data[-10240:]
                                    if np.all(last_64_channels == 0):
                                        print(f"  检测到简单补零，还原到416输入通道后重新处理")
                                        # 还原到416输入通道
                                        weight_data_416ch = weight_data[:66560]
                                        # 应用 legacy 脚本处理
                                        weight_data = process_l02_s07_wt_via_legacy_script(weight_data_416ch)
                                    else:
                                        print(f"  数据已包含非零值，保持原样")
                                else:
                                    print(f"  警告: L02_s07 原始大小 {original_size} 不符合预期 (66560或76800)，使用默认补零")
                                    weight_data = adjust_weight_size(weight_data, target_shape)
                            else:
                                # 其他层使用默认的补零方式
                                weight_data = adjust_weight_size(weight_data, target_shape)
                        else:
                            # 没有特殊处理函数，使用默认补零
                            weight_data = adjust_weight_size(weight_data, target_shape)
                    else:
                        print(f"文件 {i}: {filename} (层名: {layer_name}, 大小匹配: {original_size})")
                else:
                    print(f"文件 {i}: {filename} (层名: {layer_name}, 未找到对应的 WT shape)")
            else:
                print(f"文件 {i}: {filename} (无法提取层名, 大小: {original_size}, 未调整)")
        else:
            print(f"文件 {i}: {filename} (大小: {original_size}, 未调整)")
        
        # 保存调整后的文件
        weight_data.tofile(new_path)

    print(f"完成! 已生成 {len(sorted_files)} 个文件到文件夹: {output_folder}")


def generate_l43_wt_from_onnx(onnx_path, output_folder):
    """从 ONNX 生成 L43 WT bin 并写入输出目录（供 02_merge_wt_files 合并）。"""
    if not onnx_path or not os.path.exists(onnx_path):
        print(f"警告: 未提供有效 ONNX 路径，跳过 L43 WT 生成: {onnx_path}")
        return None

    try:
        from L43_bin_process import generate_l43_wt_bin
    except ImportError as e:
        print(f"警告: 无法导入 L43_bin_process，跳过 L43 WT 生成: {e}")
        return None

    l43_filename = "T32_T32_L43_WT.bin"
    output_bin = os.path.join(output_folder, l43_filename)
    info = generate_l43_wt_bin(onnx_path, output_bin)
    print(f"已从 ONNX 生成 L43 WT: {output_bin} ({info['bytes']} 字节)")
    return output_bin


if __name__ == '__main__':
    # 用法: python 01_rename_wt_files.py <source_folder> [output_folder] [onnx_path]
    source_folder = sys.argv[1]
    # 如果提供了第二个参数，使用它作为输出文件夹，否则使用默认路径
    if len(sys.argv) > 2:
        output_folder = sys.argv[2]
    else:
        # 默认输出文件夹：在源文件夹的父目录下创建 renamed_weights_{source_folder_name}
        source_folder_name = os.path.basename(os.path.abspath(source_folder))
        output_folder = os.path.join(os.path.dirname(os.path.abspath(source_folder)), f'renamed_weights_{source_folder_name}')

    onnx_path = sys.argv[3] if len(sys.argv) > 3 else None
    copy_and_adjust_bin_files(source_folder, output_folder)
    generate_l43_wt_from_onnx(onnx_path, output_folder)
