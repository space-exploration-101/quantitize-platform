import numpy as np
import json
import os
import struct
import sys
import re

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
    print("警告: 无法导入 layer_wt_shapes.WT_SHAPES，将使用原始 shapes")


def find_non_1024_divisible_wt(file_paths, shapes):
    """
    遍历WT文件，找出大小不能被1024整除的文件

    Args:
        file_paths: bin文件路径列表
        shapes: 每个文件对应的shape列表
    """
    non_divisible = []

    for i, (file_path, shape) in enumerate(zip(file_paths, shapes)):
        # 计算理论大小（元素数量）
        expected_size = np.prod(shape)
        # 实际文件大小（字节数，int8类型每个元素占1字节）
        try:
            file_size = os.path.getsize(file_path)
        except FileNotFoundError:
            print(f"警告：文件不存在 - {file_path}")
            continue

        # 验证文件大小是否与shape匹配
        if file_size != expected_size:
            print(f"警告：文件大小不匹配 - {file_path}，理论大小{expected_size}，实际大小{file_size}")
            continue

        # 检查是否能被1024整除
        if file_size % 1024 != 0:
            non_divisible.append({
                "index": i,
                "file_path": file_path,
                "shape": shape,
                "size": file_size,
                "remainder": file_size % 1024  # 余数
            })

    # 输出结果
    if non_divisible:
        print(f"\n发现{len(non_divisible)}个不能被1024整除的WT文件：")
        for item in non_divisible:
            print(f"索引{i}: {item['file_path']}")
            print(f"  形状: {item['shape']}")
            print(f"  大小: {item['size']}字节 (余数: {item['remainder']})")
            print(f"  1024对齐需补充: {1024 - item['remainder']}字节\n")
    else:
        print("\n所有WT文件大小均能被1024整除！")

    return non_divisible


def merge_bin_files(file_paths, shapes, output_bin='merged_weights.bin', output_meta='metadata.json',
                    output_hdr='weight_offset_table.h'):
    """
    将多个不同维度的int8 bin文件合并成一个文件

    Args:
        file_paths: bin文件路径列表
        shapes: 每个文件对应的shape列表,如 [(2,3,4,5), (1,64,128,128), ...]
        output_bin: 输出的合并bin文件路径
        output_meta: 输出的元数据json文件路径
        output_hdr: 输出的偏移量表头文件路径
    """
    all_weights = []
    metadata = {
        'dtype': 'int8',
        'total_size': 0,
        'num_files': len(file_paths),
        'weights': []
    }

    current_offset = 0
    individual_sizes_sum = 0  # 记录单个文件大小的总和
    offset_table = []  # 记录偏移量表

    for i, (file_path, shape) in enumerate(zip(file_paths, shapes)):
        # 读取int8数据
        weight = np.fromfile(file_path, dtype=np.int8)
        filename = os.path.basename(file_path)

        # 验证shape（文件应该已经在 01_rename_wt_files.py 中调整好大小）
        expected_size = np.prod(shape)
        actual_size = weight.size
        final_shape = shape
        final_weight = weight
        
        if actual_size != expected_size:
            error_msg = (
                f"错误: 文件 {i} ({os.path.basename(file_path)}) 大小不匹配！\n"
                f"  期望大小: {expected_size} (shape: {shape})\n"
                f"  实际大小: {actual_size}\n"
                f"  文件应该在 01_rename_wt_files.py 中已经调整好大小，请检查重命名步骤"
            )
            print(error_msg)
            raise ValueError(error_msg)

        # 累加单个文件大小到总和（使用最终处理后的数据大小）
        final_size = final_weight.size
        individual_sizes_sum += final_size

        # 记录元数据（使用最终 shape）
        weight_meta = {
            'index': i,
            'filename': os.path.basename(file_path),
            'shape': list(final_shape),
            'offset': current_offset,
            'size': final_size,
            'original_shape': list(shape),  # 记录原始 shape
            'original_size': actual_size    # 记录原始大小
        }
        metadata['weights'].append(weight_meta)

        # 记录偏移量表（使用最终 shape）
        # 从文件名提取层名称，例如 "T32_T32_L02_s00_conv_wt.bin" -> "L02_s00"
        match = re.search(r'L(\d+)_s(\d+)', filename)
        if match:
            layer_name = f"L{match.group(1)}_s{match.group(2)}"
        else:
            layer_name = f"weight_{i:03d}"  # 后备方案
        offset_table.append((layer_name, final_shape, current_offset, final_size))

        # 添加到列表（使用最终处理后的数据）
        all_weights.append(final_weight)
        current_offset += final_size

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(file_paths)} files")

    # 拼接所有权重
    concatenated = np.concatenate(all_weights)
    concatenated_size = concatenated.size  # 拼接后的总大小
    metadata['total_size'] = concatenated_size

    # 保存合并后的bin文件
    concatenated.tofile(output_bin)

    # 保存元数据
    with open(output_meta, 'w') as f:
        json.dump(metadata, f, indent=2)

    # 生成偏移量表头文件
    with open(output_hdr, 'w', encoding='utf-8') as hdr:
        hdr.write("// 自动生成的权重偏移表 (INT8)\n")
        hdr.write("#ifndef WEIGHT_OFFSET_TABLE_H\n")
        hdr.write("#define WEIGHT_OFFSET_TABLE_H\n\n")
        hdr.write("typedef struct { \n")
        hdr.write("    int out_channels;\n")
        hdr.write("    int in_channels; \n")
        hdr.write("    int kernel_h;\n")
        hdr.write("    int kernel_w;\n")
        hdr.write("    long offset; \n")
        hdr.write("    long size;\n")
        hdr.write("} WEIGHT_INFO;\n\n")
        hdr.write("WEIGHT_INFO weight_info[] = {\n")

        for name, shape, offset, size in offset_table:
            if len(shape) == 4:
                out_c, in_c, k_h, k_w = shape
                hdr.write(f"    {{{out_c}, {in_c}, {k_h}, {k_w}, {offset}, {size}}}, // {name}\n")
            else:
                hdr.write(f"    {{0, 0, 0, 0, {offset}, {size}}}, // {name} - 非标准4D形状: {shape}\n")

        hdr.write("};\n")
        hdr.write(f"#define TOTAL_WEIGHT_SIZE {concatenated_size}\n")
        hdr.write(f"#define NUM_WEIGHT_LAYERS {len(offset_table)}\n")
        hdr.write("#endif // WEIGHT_OFFSET_TABLE_H\n")

    # 输出详细统计信息
    print(f"\n{'=' * 60}")
    print(f"✓ 合并完成: {len(file_paths)} 个权重文件")
    print(f"{'=' * 60}")

    # 大小验证
    print(f"📊 大小统计:")
    print(f"   单个文件大小总和: {individual_sizes_sum:,} 字节")
    print(f"   拼接后总大小: {concatenated_size:,} 字节")

    if individual_sizes_sum == concatenated_size:
        print("   ✅ 验证结果: 单个文件大小总和与拼接后大小一致")
    else:
        print(f"   ⚠️ 验证警告: 大小不一致！差值为 {abs(individual_sizes_sum - concatenated_size):,} 字节")

    print(f"   总大小: {concatenated_size:,} 字节 ({concatenated_size / 1024 / 1024:.2f} MB)")

    # 输出偏移量信息
    print(f"\n📋 权重偏移量表:")
    print(f"{'索引':<6} {'名称':<15} {'形状':<20} {'偏移量':<12} {'大小':<12}")
    print(f"{'-' * 70}")

    for i, (name, shape, offset, size) in enumerate(offset_table):
        shape_str = str(shape)
        print(f"{i:<6} {name:<15} {shape_str:<20} {offset:<12} {size:<12}")

        # 每20层显示一次进度
        if (i + 1) % 20 == 0 and i + 1 < len(offset_table):
            print(f"{'-' * 70}")

    print(f"{'=' * 60}")
    print(f"📦 输出文件:")
    print(f"   合并权重文件: {output_bin}")
    print(f"   元数据文件: {output_meta}")
    print(f"   偏移量表头文件: {output_hdr}")
    print(f"{'=' * 60}")

    return metadata, offset_table

def load_weight_from_merged(merged_file, metadata, weight_index):
    """
    从合并的文件中加载特定的权重

    Args:
        merged_file: 合并后的bin文件路径
        metadata: 元数据字典或json文件路径
        weight_index: 要加载的权重索引(0-116)

    Returns:
        恢复shape后的numpy数组
    """
    # 读取元数据
    if isinstance(metadata, str):
        with open(metadata, 'r') as f:
            metadata = json.load(f)

    # 获取指定权重的信息
    weight_meta = metadata['weights'][weight_index]
    offset = weight_meta['offset']
    size = weight_meta['size']
    shape = tuple(weight_meta['shape'])

    # 读取数据
    merged = np.fromfile(merged_file, dtype=np.int8)
    weight = merged[offset:offset + size].reshape(shape)

    return weight


def verify_merged_file(bin_file, meta_file):
    """验证合并后的文件"""
    # 读取元数据
    with open(meta_file, 'r') as f:
        metadata = json.load(f)

    # 读取合并的bin文件
    merged_data = np.fromfile(bin_file, dtype=np.int8)

    print("Verification Results:")
    print(f"Metadata total_size: {metadata['total_size']}")
    print(f"Actual file size: {merged_data.size}")
    print(f"Data type: {merged_data.dtype}")

    # 验证每个权重块
    for weight_meta in metadata['weights']:
        start = weight_meta['offset']
        end = start + weight_meta['size']
        weight_slice = merged_data[start:end]

        expected_shape = tuple(weight_meta['shape'])
        expected_size = np.prod(expected_shape)

        print(f"Weight {weight_meta['index']}: offset={start}, size={weight_meta['size']}, "
              f"expected_shape={expected_shape}, actual_elements={weight_slice.size}")

        if weight_slice.size != expected_size:
            print(f"  ✗ Size mismatch!")

import sys

# 使用示例
if __name__ == "__main__":

    ################先使用文件重命名.py将WT文件命名为T32_T32_L{layer}_{sub}_conv_wt.bin格式，再运行wt文件合并.py
    folder = sys.argv[1]
    output_folder = sys.argv[2]
    file_paths = []
    #遍历文件夹中(包含子文件)所有的‘*wt.bin’文件（支持大小写）
    for root, dirs, files in os.walk(folder):
        for file in files:
            # 匹配 wt.bin 或 _WT.bin（大小写不敏感）
            if file.endswith('.bin') and ('wt.bin' in file.lower() or '_wt.bin' in file.lower()):
                file_paths.append(os.path.join(root, file))

    # 从文件名中提取层信息并排序
    def extract_layer_sort_key(filepath):
        filename = os.path.basename(filepath)
        # 优先匹配 L{数字}_s{数字} 的模式
        match = re.search(r'L(\d+)_s(\d+)', filename)
        if match:
            layer = int(match.group(1))
            sub = int(match.group(2))
            return (layer, sub)
        # 然后尝试匹配 L{数字} 这种格式（没有 s{xx}，如 L43）
        match = re.search(r'L(\d+)(?:_WT|_wt|_conv_wt)?', filename)
        if match:
            layer = int(match.group(1))
            return (layer, 0)  # L43 等层没有 sub，默认为 0，但 layer 值正确
        # 如果无法提取，尝试从文件名中提取数字（兼容旧格式）
        numbers = re.findall(r'\d+', filename)
        if len(numbers) >= 2:
            return (int(numbers[-2]), int(numbers[-1]))
        elif len(numbers) == 1:
            return (int(numbers[0]), 0)
        return (999, 0)  # 无法匹配的文件排在最后
    
    # 按层信息排序
    file_paths.sort(key=extract_layer_sort_key)
    
    ####################每一层的WT形状###################
    # 必须使用 WT_SHAPES，如果没有则报错
    if not HAS_WT_SHAPES:
        error_msg = "错误: 无法导入 layer_wt_shapes.WT_SHAPES，必须使用 WT_SHAPES 进行验证"
        print(error_msg)
        raise ImportError(error_msg)
    
    # 导入辅助函数
    try:
        from layer_wt_shapes import get_wt_shape
    except ImportError:
        error_msg = "错误: 无法导入 get_wt_shape 函数"
        print(error_msg)
        raise ImportError(error_msg)
    
    # 从文件名提取层名，构建 shapes 列表
    shapes = []
    missing_layers = []
    
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        # 从文件名提取层名（例如 "T32_T32_L02_s00_conv_wt.bin" -> "L02_s00"）
        match = re.search(r'L(\d+)_s(\d+)', filename)
        if match:
            layer = int(match.group(1))
            sub = int(match.group(2))
            layer_name = f"L{layer:02d}_s{sub:02d}"
            shape = get_wt_shape(layer_name)
            if shape:
                shapes.append(shape)
            else:
                missing_layers.append((filename, layer_name))
                # 如果找不到，使用 None 作为占位符，后续会报错
                shapes.append(None)
        else:
            # 尝试匹配 L43 这种格式（没有 s{xx}），支持 _WT 或 _wt 或 _conv_wt
            match = re.search(r'L(\d+)(?:_WT|_wt|_conv_wt)?', filename)
            if match:
                layer = int(match.group(1))
                layer_name = f"L{layer}"
                shape = get_wt_shape(layer_name)
                if shape:
                    shapes.append(shape)
                else:
                    missing_layers.append((filename, layer_name))
                    shapes.append(None)
            else:
                missing_layers.append((filename, None))
                shapes.append(None)
    
    # 检查是否有缺失的层
    if missing_layers:
        error_msg = "错误: 以下文件无法找到对应的 WT shape:\n"
        for filename, layer_name in missing_layers:
            error_msg += f"  {filename} (层名: {layer_name})\n"
        print(error_msg)
        raise ValueError(error_msg)
    
    if len(file_paths) != len(shapes):
        error_msg = (
            f"错误: 文件数量 ({len(file_paths)}) 与 shapes 数量 ({len(shapes)}) 不匹配！"
        )
        print(error_msg)
        raise ValueError(error_msg)
    
    print(f"使用 layer_wt_shapes.py 中的 WT_SHAPES (共 {len(shapes)} 个层)")
    os.makedirs(output_folder, exist_ok=True)
    merge_bin_files(file_paths, shapes, output_bin=f'{output_folder}/ALL_wt.bin', output_meta=f'{output_folder}/metadata.json',output_hdr=f'{output_folder}/weight_offset_table.h')
    find_non_1024_divisible_wt(file_paths, shapes)


    # verify_merged_file('./wt文件/ALL_wt.bin', './wt文件/metadata.json')#########验证合并后的文件size是否和合并前的一致


    # 如果你不知道shape,只知道文件大小,可以这样读取:
    # for file_path in file_paths:
    #     weight = np.fromfile(file_path, dtype=np.int8)
    #     print(f"{file_path}: size={weight.size}")
    #     # 然后你需要根据模型结构确定shape

    # 合并文件
    # metadata = merge_bin_files(file_paths, shapes)

    # 读取特定的权重
    # weight_10 = load_weight_from_merged('merged_weights.bin', 'metadata.json', 10)
    # print(f"Weight 10 shape: {weight_10.shape}")

    # print("请根据你的实际情况修改 file_paths 和 shapes 列表")


