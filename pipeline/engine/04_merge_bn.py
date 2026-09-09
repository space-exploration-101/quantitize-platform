import os
import re
import numpy as np
import sys

# 尝试导入 BN_SHAPES
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
    print("错误: 无法导入 layer_wt_shapes.BN_SHAPES，必须使用 BN_SHAPES 进行验证")
    raise


if __name__ == "__main__":
    folder_path = sys.argv[1]  # 重命名后的 BN 文件夹路径
    output_folder = sys.argv[2]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_BIN = os.path.join(output_folder, "ALL_BN.bin")
    OUTPUT_HDR = os.path.join(output_folder, "bn_offset_table.h")

    # === Step 1: 获取所有 BN 文件并排序 ===
    bn_file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.bin') and 'bn.bin' in file and 'wt.bin' not in file:
                bn_file_paths.append(os.path.join(root, file))

    if not bn_file_paths:
        print(f"错误: 在文件夹 {folder_path} 中未找到 BN 文件")
        sys.exit(1)

    # 按层名排序
    def extract_layer_sort_key(filepath):
        filename = os.path.basename(filepath)
        match = re.search(r'L(\d+)_s(\d+)', filename)
        if match:
            layer = int(match.group(1))
            sub = int(match.group(2))
            return (layer, sub)
        # 尝试匹配 L43 这种格式
        match = re.search(r'L(\d+)(?:_conv_bn)?', filename)
        if match:
            layer = int(match.group(1))
            return (layer, 0)
        return (999, 0)

    bn_file_paths.sort(key=extract_layer_sort_key)
    print(f"✅ 找到 {len(bn_file_paths)} 个 BN 文件")

    # === Step 2: 验证文件大小并合并 ===
    offset = 0
    offset_table = []
    mismatches = []

    os.makedirs(output_folder, exist_ok=True)

    with open(OUTPUT_BIN, "wb") as fout, open(OUTPUT_HDR, "w", encoding="utf-8") as hdr:
        hdr.write("// 自动生成的 BN 偏移表 (FP16)\n")
        hdr.write("#ifndef BN_OFFSET_TABLE_H\n")
        hdr.write("#define BN_OFFSET_TABLE_H\n\n")
        hdr.write("typedef struct { int C; long offset; } BN_INFO;\n")
        hdr.write("BN_INFO bn_info[] = {\n")

        for file_path in bn_file_paths:
            filename = os.path.basename(file_path)
            
            # 提取层名
            match = re.search(r'L(\d+)_s(\d+)', filename)
            if match:
                layer = int(match.group(1))
                sub = int(match.group(2))
                layer_name = f"L{layer:02d}_s{sub:02d}"
                bn_name = f"L{layer:02d}_s{sub:02d}_conv_bn"
            else:
                # 尝试匹配 L43
                match = re.search(r'L(\d+)(?:_conv_bn)?', filename)
                if match:
                    layer = int(match.group(1))
                    layer_name = f"L{layer}"
                    bn_name = f"L{layer}_conv_bn"
                else:
                    error_msg = f"错误: 无法提取层名: {filename}"
                    print(error_msg)
                    raise ValueError(error_msg)

            # 从 BN_SHAPES 获取期望的通道数
            expected_channels = get_bn_shape(layer_name)
            if not expected_channels:
                error_msg = f"错误: 无法在 BN_SHAPES 中找到层 {layer_name} (文件: {filename})"
                print(error_msg)
                raise ValueError(error_msg)

            # 读取源文件
            with open(file_path, "rb") as fin:
                data = fin.read()

            # 验证文件大小是否正确
            expected_size = expected_channels * 2 * 2  # channels * 2 params * 2 bytes
            actual_size = len(data)

            if actual_size != expected_size:
                error_msg = (
                    f"错误: 文件 {filename} 大小不匹配！\n"
                    f"  层名: {layer_name}\n"
                    f"  期望通道数: {expected_channels}\n"
                    f"  期望大小: {expected_size} 字节 (channels * 2 * 2)\n"
                    f"  实际大小: {actual_size} 字节\n"
                    f"  文件应该在 03_rename_bn_files.py 中已经调整好大小，请检查重命名步骤"
                )
                print(error_msg)
                raise ValueError(error_msg)

            # 写入合并文件
            fout.write(data)

            # 记录偏移量
            hdr.write(f"    {{{expected_channels}, {offset}}}, // {bn_name}\n")
            offset_table.append((bn_name, expected_channels, offset))
            offset += actual_size

        hdr.write("};\n")
        hdr.write(f"#define TOTAL_BN_SIZE {offset}\n")
        hdr.write("#endif // BN_OFFSET_TABLE_H\n")

    print(f"✅ 拼接完成，共 {len(offset_table)} 层")
    print(f"📦 输出文件: {OUTPUT_BIN}")
    print(f"🧾 偏移表: {OUTPUT_HDR}")

    # === Step 3: 输出统计和验证 ===
    total_size = os.path.getsize(OUTPUT_BIN)
    print(f"📊 ALL_BN.bin 总大小: {total_size / 1024:.2f} KB")
    print(f"🔢 总偏移量: {offset} 字节")
