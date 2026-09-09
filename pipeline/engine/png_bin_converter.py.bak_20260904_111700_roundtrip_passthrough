#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PNG与BIN转换工具
功能：
1. PNG转BIN: 读取PNG图片，resize到2000x2000，转为灰度，将8bit线性映射到12bit(0..4095)后按行打包
2. BIN转PNG: 读取BIN文件，将12bit(0..4095)线性映射到8bit(0..255)后存PNG
3. 精度损失验证: 比较原始图片和转换后的图片
"""

import os
import sys
import argparse
import tempfile
import shutil
import numpy as np
import cv2
from typing import Tuple, Optional


def _imread_unicode(path: str, flags=int(cv2.IMREAD_COLOR)):
    """支持中文等 Unicode 路径的图片读取（Windows 下 cv2.imread 会损坏）"""
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    if img is None:
        raise ValueError(f"无法解码图片: {path}")
    return img


def _imwrite_unicode(path: str, img: np.ndarray):
    """支持中文等 Unicode 路径的图片保存（先写临时文件再移动，避免 Windows 中文路径损坏）"""
    path = os.path.abspath(path)
    ext = os.path.splitext(path)[1].lower() or '.png'
    if ext not in ['.png', '.jpg', '.jpeg']:
        ext = '.png'
    success, buf = cv2.imencode(ext, img)
    if not success:
        raise ValueError("图片编码失败")
    # 先写到临时文件（纯 ASCII 路径），再移动到目标路径，避免中文路径导致文件损坏
    fd, tmp_path = tempfile.mkstemp(suffix=ext, prefix='img_')
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(buf.tobytes())
        shutil.move(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        raise


def _u12_to_u8(data_12bit: np.ndarray) -> np.ndarray:
    """12bit 满量程 0..4095 线性缩放到 8bit 0..255（四舍五入）。"""
    x = data_12bit.astype(np.float64) * (255.0 / 4095.0)
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


def _u8_to_u12(gray_u8: np.ndarray) -> np.ndarray:
    """8bit 0..255 线性映射到 12bit 0..4095，与 _u12_to_u8 配对用于 png2bin。"""
    x = gray_u8.astype(np.float64) * (4095.0 / 255.0)
    return np.clip(np.round(x), 0, 4095).astype(np.uint16)


def pack_12bit_to_bytes(data_12bit: np.ndarray) -> bytes:
    """
    将12bit值打包成字节数组
    每2个12bit值打包成3个字节（2*12=24=3*8）
    
    Args:
        data_12bit: 12bit值数组（uint16类型，但值范围0-4095）
    
    Returns:
        打包后的字节数组
    """
    values = np.asarray(data_12bit, dtype=np.uint16).reshape(-1) & np.uint16(0x0FFF)
    if values.size % 2:
        values = np.pad(values, (0, 1), mode="constant")
    pairs = values.reshape(-1, 2)
    val1 = pairs[:, 0]
    val2 = pairs[:, 1]
    packed = np.empty((pairs.shape[0], 3), dtype=np.uint8)
    packed[:, 0] = val1 & np.uint16(0xFF)
    packed[:, 1] = ((val2 & np.uint16(0x0F)) << np.uint16(4)) | (val1 >> np.uint16(8))
    packed[:, 2] = val2 >> np.uint16(4)
    return packed.tobytes()


def unpack_12bit_from_bytes(packed_data: bytes) -> np.ndarray:
    """
    从字节数组解包12bit值
    
    Args:
        packed_data: 打包的字节数组（每3字节包含2个12bit值）
    
    Returns:
        12bit值数组（uint16类型）
    """
    raw = np.frombuffer(packed_data, dtype=np.uint8)
    if raw.size % 3:
        raise ValueError("12bit packed data length must be divisible by 3")
    triples = raw.reshape(-1, 3).astype(np.uint16)
    unpacked = np.empty(triples.shape[0] * 2, dtype=np.uint16)
    unpacked[0::2] = ((triples[:, 1] & 0x0F) << 8) | triples[:, 0]
    unpacked[1::2] = (triples[:, 2] << 4) | (triples[:, 1] >> 4)
    return unpacked


def png_to_bin(png_path: str, bin_path: str, target_size: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
    """
    将PNG图片转换为BIN文件
    
    Args:
        png_path: 输入PNG文件路径
        bin_path: 输出BIN文件路径
        target_size: 目标尺寸（默认2000x2000）
    
    Returns:
        (原始图片数组, 转换后的图片数组)
    """
    print(f"正在读取PNG文件: {png_path}")
    
    # 读取PNG图片（使用支持中文路径的读取方式）
    img = _imread_unicode(png_path)
    
    print(f"原始图片尺寸: {img.shape}")
    
    # 转为灰度图
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    print(f"灰度图尺寸: {gray.shape}")
    print(f"灰度图数据类型: {gray.dtype}")
    print(f"灰度图值范围: [{np.min(gray)}, {np.max(gray)}]")
    
    # Resize到目标尺寸
    gray_resized = cv2.resize(gray, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    print(f"Resize后尺寸: {gray_resized.shape}")
    
    # 保存原始数组用于后续比较
    original_array = gray_resized.copy()
    
    # 8bit -> 12bit：满量程线性映射（非左移4位），与 bin2png 的 12->8 缩放互逆
    gray_12bit = _u8_to_u12(gray_resized)
    
    print(f"12bit值范围: [{np.min(gray_12bit)}, {np.max(gray_12bit)}]")
    
    # 按行存储，每行4096字节，不够的补零
    BYTES_PER_ROW = 4096
    height, width = gray_12bit.shape
    
    print(f"图片尺寸: {height}行 × {width}列")
    
    # 计算每行需要的有效字节数（每2个像素打包成3字节）
    bytes_per_row_data = (width + 1) // 2 * 3  # 每行实际需要的字节数
    print(f"每行有效字节数: {bytes_per_row_data} 字节")
    print(f"每行固定大小: {BYTES_PER_ROW} 字节")
    
    if bytes_per_row_data > BYTES_PER_ROW:
        raise ValueError(f"每行需要的字节数 ({bytes_per_row_data}) 超过了行大小限制 ({BYTES_PER_ROW})")
    
    # 整图向量化打包，保持逐行 4096 字节布局与旧实现逐字节一致。
    values = gray_12bit
    if width % 2:
        values = np.pad(values, ((0, 0), (0, 1)), mode="constant")
    pairs = values.reshape(height, -1, 2)
    val1 = pairs[:, :, 0]
    val2 = pairs[:, :, 1]
    triples = np.empty((height, pairs.shape[1], 3), dtype=np.uint8)
    triples[:, :, 0] = val1 & np.uint16(0xFF)
    triples[:, :, 1] = ((val2 & np.uint16(0x0F)) << np.uint16(4)) | (val1 >> np.uint16(8))
    triples[:, :, 2] = val2 >> np.uint16(4)
    packed_rows = np.zeros((height, BYTES_PER_ROW), dtype=np.uint8)
    packed_rows[:, :bytes_per_row_data] = triples.reshape(height, bytes_per_row_data)
    packed_rows.tofile(bin_path)
    total_valid_bytes = height * bytes_per_row_data
    
    file_size = os.path.getsize(bin_path)
    expected_file_size = height * BYTES_PER_ROW
    print(f"✅ BIN文件已保存: {bin_path}")
    print(f"   总行数: {height}")
    print(f"   每行大小: {BYTES_PER_ROW} 字节")
    print(f"   每行有效字节数: {bytes_per_row_data} 字节")
    print(f"   总有效字节数: {total_valid_bytes} 字节")
    print(f"   实际文件大小: {file_size} 字节 (期望: {expected_file_size} 字节)")
    print(f"   总填充字节数: {file_size - total_valid_bytes} 字节")
    
    return original_array, gray_12bit


def bin_to_png(bin_path: str, png_path: str, width: int = 2000, height: int = 2000) -> np.ndarray:
    """
    将BIN文件转换为PNG图片
    
    Args:
        bin_path: 输入BIN文件路径
        png_path: 输出PNG文件路径
        width: 图片宽度
        height: 图片高度
    
    Returns:
        还原后的图片数组
    """
    print(f"正在读取BIN文件: {bin_path}")
    
    BYTES_PER_ROW = 4096
    file_size = os.path.getsize(bin_path)
    
    # 计算预期的文件大小（每行4096字节）
    expected_file_size = height * BYTES_PER_ROW
    
    if file_size != expected_file_size:
        print(f"⚠️  警告: 文件大小不匹配")
        print(f"   预期: {expected_file_size} 字节 ({height}行 × {BYTES_PER_ROW}字节/行)")
        print(f"   实际: {file_size} 字节")
        # 尝试从文件大小推断行数
        inferred_height = file_size // BYTES_PER_ROW
        if file_size % BYTES_PER_ROW == 0:
            height = inferred_height
            print(f"   自动推断行数: {height}")
        else:
            raise ValueError(f"文件大小不是 {BYTES_PER_ROW} 字节的整数倍")
    
    # 计算每行的有效字节数
    bytes_per_row_data = (width + 1) // 2 * 3  # 每行实际需要的字节数
    print(f"图片尺寸: {height}行 × {width}列")
    print(f"每行固定大小: {BYTES_PER_ROW} 字节")
    print(f"每行有效字节数: {bytes_per_row_data} 字节")
    
    if bytes_per_row_data > BYTES_PER_ROW:
        raise ValueError(f"每行需要的字节数 ({bytes_per_row_data}) 超过了行大小限制 ({BYTES_PER_ROW})")
    
    # 整图向量化解包。
    raw = np.fromfile(bin_path, dtype=np.uint8)
    if raw.size != height * BYTES_PER_ROW:
        raise ValueError(f"BIN 数据长度不匹配: {raw.size}")
    valid = raw.reshape(height, BYTES_PER_ROW)[:, :bytes_per_row_data]
    triples = valid.reshape(height, -1, 3).astype(np.uint16)
    unpacked = np.empty((height, triples.shape[1] * 2), dtype=np.uint16)
    unpacked[:, 0::2] = ((triples[:, :, 1] & 0x0F) << 8) | triples[:, :, 0]
    unpacked[:, 1::2] = (triples[:, :, 2] << 4) | (triples[:, :, 1] >> 4)
    data_12bit_2d = unpacked[:, :width]
    total_valid_bytes = height * bytes_per_row_data

    # 统计每行有效字节数
    print(f"\n每行有效字节数统计:")
    print(f"  期望每行有效字节数: {bytes_per_row_data} 字节")
    print(f"  ✅ 所有行的有效字节数都正确")
    
    print(f"\n总有效字节数: {total_valid_bytes} 字节")
    print(f"总填充字节数: {file_size - total_valid_bytes} 字节")
    print(f"填充比例: {(file_size - total_valid_bytes) / file_size * 100:.2f}%")
    
    data_12bit = data_12bit_2d.reshape(-1)
    
    print(f"解包后数据长度: {len(data_12bit)}")
    print(f"12bit值范围: [{np.min(data_12bit)}, {np.max(data_12bit)}]")
    
    # 12bit -> 8bit：满量程线性缩放（避免直接截断低4bit）
    data_8bit = _u12_to_u8(data_12bit)
    
    print(f"还原后8bit值范围: [{np.min(data_8bit)}, {np.max(data_8bit)}]")
    
    # 重塑为2D数组（按行存储）
    img_restored = data_8bit.reshape((height, width))
    
    print(f"还原后图片尺寸: {img_restored.shape}")
    
    # 保存图片（使用支持中文路径的保存方式，避免 Windows 下损坏）
    _imwrite_unicode(png_path, img_restored)
    
    print(f"✅ PNG文件已保存: {png_path}")
    
    return img_restored


def verify_accuracy(original: np.ndarray, restored: np.ndarray) -> dict:
    """
    验证转换精度损失
    
    Args:
        original: 原始图片数组（uint8, 0-255）
        restored: 还原后的图片数组（uint8, 0-255）
    
    Returns:
        包含各种精度指标的字典
    """
    print("\n" + "="*60)
    print("精度损失验证")
    print("="*60)
    
    # 确保两个数组形状相同
    if original.shape != restored.shape:
        raise ValueError(f"数组形状不匹配: {original.shape} vs {restored.shape}")
    
    # 计算差异
    diff = original.astype(np.int16) - restored.astype(np.int16)
    
    # 统计指标
    mse = np.mean(diff ** 2)
    mae = np.mean(np.abs(diff))
    max_error = np.max(np.abs(diff))
    min_error = np.min(np.abs(diff))
    
    # 计算PSNR
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    
    # 计算完全相同的像素比例
    identical_pixels = np.sum(original == restored)
    identical_ratio = identical_pixels / original.size * 100
    
    # 计算误差分布
    error_0 = np.sum(np.abs(diff) == 0)
    error_1 = np.sum(np.abs(diff) == 1)
    error_2 = np.sum(np.abs(diff) == 2)
    error_3_plus = np.sum(np.abs(diff) >= 3)
    
    results = {
        'mse': mse,
        'mae': mae,
        'max_error': int(max_error),
        'min_error': int(min_error),
        'psnr': psnr,
        'identical_pixels': int(identical_pixels),
        'identical_ratio': identical_ratio,
        'total_pixels': int(original.size),
        'error_distribution': {
            'error_0': int(error_0),
            'error_1': int(error_1),
            'error_2': int(error_2),
            'error_3_plus': int(error_3_plus)
        }
    }
    
    # 打印结果
    print(f"均方误差 (MSE): {mse:.6f}")
    print(f"平均绝对误差 (MAE): {mae:.6f}")
    print(f"最大误差: {max_error}")
    print(f"最小误差: {min_error}")
    print(f"峰值信噪比 (PSNR): {psnr:.2f} dB")
    print(f"\n像素统计:")
    print(f"  总像素数: {original.size:,}")
    print(f"  完全相同像素: {identical_pixels:,} ({identical_ratio:.2f}%)")
    print(f"\n误差分布:")
    print(f"  误差 = 0: {error_0:,} ({error_0/original.size*100:.2f}%)")
    print(f"  误差 = 1: {error_1:,} ({error_1/original.size*100:.2f}%)")
    print(f"  误差 = 2: {error_2:,} ({error_2/original.size*100:.2f}%)")
    print(f"  误差 ≥ 3: {error_3_plus:,} ({error_3_plus/original.size*100:.2f}%)")
    print("="*60)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='PNG与BIN转换工具')
    parser.add_argument('mode', choices=['png2bin', 'bin2png', 'verify'], 
                       help='转换模式: png2bin(PNG转BIN), bin2png(BIN转PNG), verify(验证精度)')
    parser.add_argument('input', help='输入文件路径')
    parser.add_argument('output', help='输出文件路径')
    parser.add_argument('--size', type=int, default=2000, 
                       help='目标尺寸（默认2000x2000）')
    parser.add_argument('--width', type=int, default=2000,
                       help='BIN转PNG时的图片宽度（默认2000）')
    parser.add_argument('--height', type=int, default=2000,
                       help='BIN转PNG时的图片高度（默认2000）')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'png2bin':
            print("="*60)
            print("PNG转BIN")
            print("="*60)
            original, converted = png_to_bin(args.input, args.output, args.size)
            print(f"\n✅ 转换完成!")
            print(f"   输入: {args.input}")
            print(f"   输出: {args.output}")
            
        elif args.mode == 'bin2png':
            print("="*60)
            print("BIN转PNG")
            print("="*60)
            restored = bin_to_png(args.input, args.output, args.width, args.height)
            print(f"\n✅ 转换完成!")
            print(f"   输入: {args.input}")
            print(f"   输出: {args.output}")
            
        elif args.mode == 'verify':
            print("="*60)
            print("精度验证")
            print("="*60)
            # 验证模式：需要原始PNG和还原后的PNG
            if not args.input.endswith('.png') or not args.output.endswith('.png'):
                print("错误: 验证模式需要两个PNG文件")
                return 1
            
            original_img = _imread_unicode(args.input, cv2.IMREAD_GRAYSCALE)
            
            # 如果原始图片不是2000x2000，先resize
            if original_img.shape != (args.size, args.size):
                original_img = cv2.resize(original_img, (args.size, args.size))
            
            restored_img = _imread_unicode(args.output, cv2.IMREAD_GRAYSCALE)
            
            results = verify_accuracy(original_img, restored_img)
            print(f"\n✅ 验证完成!")
            
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
