import os
import re
import sys
import numpy as np

# 尝试导入正确的 shapes
try:
    # 添加 bin_process 目录到路径
    bin_process_dir = os.path.join(os.path.dirname(__file__), '..', 'bin_process')
    if bin_process_dir not in sys.path:
        sys.path.insert(0, bin_process_dir)
    from layer_wt_shapes import BN_SHAPES, get_bn_shape
    HAS_BN_SHAPES = True
except ImportError:
    BN_SHAPES = None
    HAS_BN_SHAPES = False
    print("警告: 无法导入 layer_wt_shapes.BN_SHAPES，将无法进行大小调整")

# 导入特殊层处理函数
try:
    from special_layer_processors import process_l02_s00_bn
    HAS_SPECIAL_PROCESSORS = True
except ImportError:
    HAS_SPECIAL_PROCESSORS = False
    print("警告: 无法导入特殊层处理函数，将使用默认的补零方式")


def extract_layer_info(filename):
    """
    从文件名中提取层信息 L{layer}_{sub}
    
    Args:
        filename: 文件名，例如 "T32_T32_L02_s00_conv_bn.bin" 或 "L02_s00_conv_bn.bin"
    
    Returns:
        (layer, sub) 元组，例如 (2, 0)，如果无法提取则返回 None
    """
    # 匹配 L{数字}_s{数字} 的模式
    match = re.search(r'L(\d+)_s(\d+)', filename)
    if match:
        layer = int(match.group(1))
        sub = int(match.group(2))
        return (layer, sub)
    return None


def adjust_bn_size(bn_data, target_channels):
    """
    调整 BN 数据的大小以匹配目标通道数
    
    Args:
        bn_data: numpy数组，uint8类型（FP16格式，每个参数2字节）
        target_channels: 目标通道数
    
    Returns:
        调整后的 BN 数据
    """
    # BN 文件大小 = channels * 2 * 2（每个通道2个参数，每个参数2字节）
    target_size = target_channels * 2 * 2
    actual_size = len(bn_data)
    
    if actual_size == target_size:
        return bn_data
    elif actual_size < target_size:
        # 需要补零（FP16格式，补0x0000，即两个字节的0）
        padding_size = target_size - actual_size
        padding = np.zeros(padding_size, dtype=np.uint8)
        adjusted = np.concatenate([bn_data, padding])
        print(f"  补零: {actual_size} -> {target_size} (+{padding_size} 字节, +{padding_size//4} 个参数)")
        return adjusted
    else:
        # 需要截取
        adjusted = bn_data[:target_size]
        print(f"  截取: {actual_size} -> {target_size} (-{actual_size - target_size} 字节, -{(actual_size - target_size)//4} 个参数)")
        return adjusted


def copy_and_adjust_bn_files(source_folder, output_folder):
    """
    复制 BN 文件到新文件夹，调整大小以匹配正确的通道数，保持原有文件名不变
    
    Args:
        source_folder: 源文件夹路径
        output_folder: 输出文件夹路径
    """
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取所有 BN bin 文件
    file_paths = []
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith('.bin') and 'bn.bin' in file and 'wt.bin' not in file:
                file_paths.append(os.path.join(root, file))
    
    if not file_paths:
        print(f"在文件夹 {source_folder} 中未找到 BN bin 文件")
        return
    
    print(f"找到 {len(file_paths)} 个 BN bin 文件")
    
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
    
    if not HAS_BN_SHAPES:
        print("警告: 无法导入 BN_SHAPES，将不进行大小调整")
    
    # 复制、调整大小，保持原有文件名
    for i, source_path in enumerate(sorted_files):
        filename = os.path.basename(source_path)
        # 保持原有文件名，不进行重命名
        new_path = os.path.join(output_folder, filename)
        
        # 读取原始文件（FP16格式）
        bn_data_fp16 = np.fromfile(source_path, dtype=np.float16)
        original_size = len(bn_data_fp16) * 2  # FP16每个元素2字节
        original_elements = len(bn_data_fp16)
        original_channels = original_elements // 2
        
        # 如果可以使用 BN_SHAPES，调整大小
        if HAS_BN_SHAPES:
            # 从文件名提取层名（例如 "T32_T32_L02_s00_conv_bn.bin" -> "L02_s00"）
            layer_info = extract_layer_info(filename)
            if layer_info:
                layer, sub = layer_info
                layer_name = f"L{layer:02d}_s{sub:02d}"
                target_channels = get_bn_shape(layer_name)
                
                if target_channels:
                    target_size = target_channels * 2 * 2  # channels * 2 params * 2 bytes
                    target_elements = target_channels * 2  # channels * 2 params
                    
                    if original_size != target_size:
                        print(f"文件 {i}: {filename}")
                        print(f"  层名: {layer_name}")
                        print(f"  原始大小: {original_size} 字节 ({original_elements} 个参数, {original_channels} 通道)")
                        print(f"  目标通道数: {target_channels}")
                        print(f"  目标大小: {target_size} 字节 ({target_elements} 个参数)")
                        
                        # 检查是否是特殊层，需要使用特殊处理
                        if HAS_SPECIAL_PROCESSORS:
                            if layer_name == "L02_s00":
                                # L02_s00: 从160通道扩展到192通道
                                if original_elements == 320:  # 160通道 * 2参数
                                    print(f"  使用特殊处理: L02_s00 (160 -> 192 通道)")
                                    bn_data_fp16 = process_l02_s00_bn(bn_data_fp16)
                                    # 转换为uint8格式保存
                                    bn_data_uint8 = bn_data_fp16.view(np.uint8)
                                elif original_elements == 384:  # 已经是192通道，但可能是简单补零的
                                    print(f"  检测到192通道数据，检查是否需要重新处理...")
                                    # 检查后32通道是否全为0（简单补零的特征）
                                    last_32_channels = bn_data_fp16[-64:]  # 最后32通道 * 2参数
                                    if np.all(last_32_channels == 0):
                                        print(f"  检测到简单补零，还原到160通道后重新处理")
                                        # 还原到160通道
                                        bn_data_160ch = bn_data_fp16[:320]
                                        # 应用特殊处理
                                        bn_data_fp16 = process_l02_s00_bn(bn_data_160ch)
                                        bn_data_uint8 = bn_data_fp16.view(np.uint8)
                                    else:
                                        print(f"  数据已包含非零值，保持原样")
                                        bn_data_uint8 = bn_data_fp16.view(np.uint8)
                                else:
                                    print(f"  警告: L02_s00 原始大小 {original_elements} 个参数不符合预期 (320或384)，使用默认补零")
                                    bn_data_uint8 = np.fromfile(source_path, dtype=np.uint8)
                                    bn_data_uint8 = adjust_bn_size(bn_data_uint8, target_channels)
                            else:
                                # 其他层使用默认的补零方式
                                bn_data_uint8 = np.fromfile(source_path, dtype=np.uint8)
                                bn_data_uint8 = adjust_bn_size(bn_data_uint8, target_channels)
                        else:
                            # 没有特殊处理函数，使用默认补零
                            bn_data_uint8 = np.fromfile(source_path, dtype=np.uint8)
                            bn_data_uint8 = adjust_bn_size(bn_data_uint8, target_channels)
                    else:
                        print(f"文件 {i}: {filename} (层名: {layer_name}, 大小匹配: {original_size} 字节)")
                        bn_data_uint8 = np.fromfile(source_path, dtype=np.uint8)
                else:
                    print(f"文件 {i}: {filename} (层名: {layer_name}, 未找到对应的 BN shape)")
                    bn_data_uint8 = np.fromfile(source_path, dtype=np.uint8)
            else:
                print(f"文件 {i}: {filename} (无法提取层名, 大小: {original_size}, 未调整)")
                bn_data_uint8 = np.fromfile(source_path, dtype=np.uint8)
        else:
            print(f"文件 {i}: {filename} (大小: {original_size}, 未调整)")
            bn_data_uint8 = np.fromfile(source_path, dtype=np.uint8)
        
        # 保存调整后的文件
        bn_data_uint8.tofile(new_path)
    
    print(f"完成! 已生成 {len(sorted_files)} 个文件到文件夹: {output_folder}")


if __name__ == '__main__':
    # 使用示例
    source_folder = sys.argv[1]
    # 如果提供了第二个参数，使用它作为输出文件夹，否则使用默认路径
    if len(sys.argv) > 2:
        output_folder = sys.argv[2]
    else:
        # 默认输出文件夹：在源文件夹的父目录下创建 renamed_bn_{source_folder_name}
        source_folder_name = os.path.basename(os.path.abspath(source_folder))
        output_folder = os.path.join(os.path.dirname(os.path.abspath(source_folder)), f'renamed_bn_{source_folder_name}')
    copy_and_adjust_bn_files(source_folder, output_folder)
