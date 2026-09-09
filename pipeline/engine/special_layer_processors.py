"""
特殊层的处理函数：从原始通道数重新排列到目标通道数
这些函数实现了与 L02_s00_bn_bin_process.py、L02_s00_wt_bin_process.py、L02_s07_wt_bin_process.py 相同的逻辑
"""
import numpy as np


def process_l02_s00_bn(bn_data_160ch):
    """
    处理 L02_s00 BN：从160通道重新排列到192通道
    
    规则：
    - 通道 0-79: 直接复制原通道 0-79
    - 通道 80-95: 填0
    - 通道 96-175: 从原通道 80-159 复制（即原通道索引+16）
    - 通道 176-191: 填0
    
    Args:
        bn_data_160ch: numpy数组，FP16格式，160通道 (320个元素)
    
    Returns:
        numpy数组，FP16格式，192通道 (384个元素)
    """
    # 确保输入是160通道
    if len(bn_data_160ch) != 320:  # 160通道 * 2参数
        raise ValueError(f"期望160通道BN数据（320个元素），实际得到 {len(bn_data_160ch)} 个元素")
    
    # 重塑为 (160, 2) - 每个通道有2个参数（bias, scale）
    fp_bn = bn_data_160ch.reshape(160, 2)
    
    # 创建新的192通道BN
    new_fp_bn = np.zeros((192, 2), dtype=np.float16)
    
    for i in range(192):
        for j in range(2):
            if i < 80:
                # 通道 0-79: 直接复制
                new_fp_bn[i, j] = fp_bn[i, j]
            elif i < 96:
                # 通道 80-95: 填0
                new_fp_bn[i, j] = 0
            elif i < 176:
                # 通道 96-175: 从原通道 80-159 复制（即原通道索引+16）
                new_fp_bn[i, j] = fp_bn[i - 16, j]
            else:
                # 通道 176-191: 填0
                new_fp_bn[i, j] = 0
    
    # 展平返回
    return new_fp_bn.flatten()


def process_l02_s00_wt(wt_data_160ch):
    """
    处理 L02_s00 WT：从160输出通道重新排列到192输出通道
    
    规则：
    - 输出通道 0-79: 直接复制原输出通道 0-79
    - 输出通道 80-95: 填0
    - 输出通道 96-175: 从原输出通道 80-159 复制（即原通道索引+16）
    - 输出通道 176-191: 填0
    
    Args:
        wt_data_160ch: numpy数组，int8格式，形状应该是 (160, 160, 1, 1) 展平 = 25600个元素
    
    Returns:
        numpy数组，int8格式，形状 (192, 160, 1, 1) 展平 = 30720个元素
    """
    # 确保输入是160输出通道
    if len(wt_data_160ch) != 25600:  # 160 * 160 * 1 * 1
        raise ValueError(f"期望160输出通道WT数据（25600个元素），实际得到 {len(wt_data_160ch)} 个元素")
    
    # 重塑为 (160, 160, 1, 1)
    int_weight = wt_data_160ch.reshape(160, 160, 1, 1)
    
    # 创建新的192输出通道WT
    new_int_weight = np.zeros((192, 160, 1, 1), dtype=np.int8)
    
    for i in range(192):
        for j in range(160):
            if i < 80:
                # 输出通道 0-79: 直接复制
                new_int_weight[i, j, 0, 0] = int_weight[i, j, 0, 0]
            elif i < 96:
                # 输出通道 80-95: 填0
                new_int_weight[i, j, 0, 0] = 0
            elif i < 176:
                # 输出通道 96-175: 从原输出通道 80-159 复制
                new_int_weight[i, j, 0, 0] = int_weight[i - 16, j, 0, 0]
            else:
                # 输出通道 176-191: 填0
                new_int_weight[i, j, 0, 0] = 0
    
    # 展平返回
    return new_int_weight.flatten()


def process_l02_s07_wt(wt_data_416ch):
    """
    处理 L02_s07 WT：从416输入通道重新排列到480输入通道
    
    规则（对输入通道）：
    - 输入通道 0-79: 直接复制原输入通道 0-79
    - 输入通道 80-95: 填0
    - 输入通道 96-175: 从原输入通道 80-159 复制
    - 输入通道 176-191: 填0
    - 输入通道 192-271: 从原输入通道 160-239 复制
    - 输入通道 272-287: 填0
    - 输入通道 288-367: 从原输入通道 240-319 复制
    - 输入通道 368-383: 填0
    - 输入通道 384-463: 从原输入通道 320-399 复制
    - 输入通道 464-479: 填0
    
    Args:
        wt_data_416ch: numpy数组，int8格式，形状应该是 (160, 416, 1, 1) 展平 = 66560个元素
    
    Returns:
        numpy数组，int8格式，形状 (160, 480, 1, 1) 展平 = 76800个元素
    """
    # 确保输入是416输入通道
    if len(wt_data_416ch) != 66560:  # 160 * 416 * 1 * 1
        raise ValueError(f"期望416输入通道WT数据（66560个元素），实际得到 {len(wt_data_416ch)} 个元素")
    
    # 重塑为 (160, 416, 1, 1)
    int_weight = wt_data_416ch.reshape(160, 416, 1, 1)
    
    # 创建新的480输入通道WT
    new_int_weight = np.zeros((160, 480, 1, 1), dtype=np.int8)
    
    for i in range(160):  # 输出通道
        for j in range(480):  # 输入通道
            if j < 80:
                # 输入通道 0-79: 直接复制
                new_int_weight[i, j, 0, 0] = int_weight[i, j, 0, 0]
            elif j < 96:
                # 输入通道 80-95: 填0
                new_int_weight[i, j, 0, 0] = 0
            elif j < 96 + 80:  # j < 176
                # 输入通道 96-175: 从原输入通道 80-159 复制
                new_int_weight[i, j, 0, 0] = int_weight[i, j - 16, 0, 0]
            elif j < 96 + 96:  # j < 192
                # 输入通道 176-191: 填0
                new_int_weight[i, j, 0, 0] = 0
            elif j < 96*2 + 80:  # j < 272
                # 输入通道 192-271: 从原输入通道 160-239 复制
                new_int_weight[i, j, 0, 0] = int_weight[i, j - 16*2, 0, 0]
            elif j < 96*2 + 96:  # j < 288
                # 输入通道 272-287: 填0
                new_int_weight[i, j, 0, 0] = 0
            elif j < 96*3 + 80:  # j < 368
                # 输入通道 288-367: 从原输入通道 240-319 复制
                new_int_weight[i, j, 0, 0] = int_weight[i, j - 16*3, 0, 0]
            elif j < 96*3 + 96:  # j < 384
                # 输入通道 368-383: 填0
                new_int_weight[i, j, 0, 0] = 0
            elif j < 96*4 + 80:  # j < 464
                # 输入通道 384-463: 从原输入通道 320-399 复制
                new_int_weight[i, j, 0, 0] = int_weight[i, j - 16*4, 0, 0]
            else:
                # 输入通道 464-479: 填0
                new_int_weight[i, j, 0, 0] = 0
    
    # 展平返回
    return new_int_weight.flatten()
