
import sys
import os

# 清理 sys.path，移除 Isaac Sim 等可能冲突的路径
# 这必须在导入 numba 之前完成
sys.path = [p for p in sys.path if p and 'isaac-sim' not in p.lower()]

import struct
import numpy as np
import random
import numba as nb
from numba import jit

#################### hardware defines ##########################
base_Tin                    =32
Tout                        =32

Tin                         =base_Tin
Tb                          =1

def parallel_of_feature(input, CH, H, W):
    slice_of_CH = (CH + Tout - 1) // Tout
    src = np.asarray(input).reshape(CH, H, W)
    padded = np.zeros((slice_of_CH * Tout, H, W), dtype=src.dtype)
    padded[:CH] = src
    return padded.reshape(slice_of_CH, Tout, H, W).transpose(0, 2, 3, 1)

def parallel_of_bn(bn_input, CH):
    slice_of_CH = (CH + Tout - 1) // Tout
    Total_CH = slice_of_CH*Tout*2
    src = np.asarray(bn_input)
    bn_output = np.zeros(Total_CH, dtype=src.dtype)
    bn_output[: CH * 2] = src[:CH, :2].reshape(-1)
    return bn_output


def parallel_weight(weight_input, CHout, CHin, Ky, Kx):
    slice_of_CHin = (CHin + Tin - 1) // Tin
    slice_of_CHout = (CHout + Tout - 1) // Tout
    src = np.asarray(weight_input).reshape(CHout, CHin, Ky, Kx)
    padded = np.zeros(
        (slice_of_CHout * Tout, slice_of_CHin * Tin, Ky, Kx), dtype=src.dtype
    )
    padded[:CHout, :CHin] = src
    return padded.reshape(
        slice_of_CHout, Tout, slice_of_CHin, Tin, Ky, Kx
    ).transpose(0, 2, 4, 5, 1, 3)


# @jit(nopython=True)
def fp16int_mul_conv_bn(conv_input, conv_weight, bn_wt_and_bias, conv_output, Ky, Kx, Sy, Sx, Py, Px):

    CHin = conv_input.shape[0]
    Hin = conv_input.shape[1]
    Win = conv_input.shape[2]
    CHout = conv_output.shape[0]
    Hout = conv_output.shape[1]
    Wout = conv_output.shape[2]

    for chout in range(CHout):
        for hout in range(Hout):
            for wout in range(Wout):
                tp_sum = 0
                for chin in range(CHin):
                    for ky in range(Ky):
                        for kx in range(Kx):
                            hin = hout*Sy - Py + ky
                            win = wout*Sx - Px + kx
                            if not (hin < 0 or win < 0 or hin >= Hin or win >= Win):
                                dat = conv_input[chin][hin][win]
                            else:
                                dat = 0
                            wt = conv_weight[chout][chin][ky][kx]
                            tp_sum += dat * wt

                conv_output[chout][hout][wout] = tp_sum * bn_wt_and_bias[2*chout+1] + bn_wt_and_bias[2*chout+0]
                # print("output[ch%d][h%d][w%d]=%10f" %(chout, hout, wout, conv_output[chout][hout][wout]))
    return conv_output


def fp16array_to_bin(input_array, output_file):
    float_array = np.array(input_array, dtype=np.float32) # 转换为numpy数组
    fp16_array = float_array.astype(np.float16)          # 转换为FP16格式
    fp16_array.tofile(output_file)                       # 保存为二进制文件


def INTarray_to_bin(dat_in, bin_out):
    values = np.asarray(dat_in).reshape(-1).astype(np.int16, copy=False)
    if values.size % 2:
        print("txt行数不是2的倍数，补充 1 行0")
        values = np.pad(values, (0, 1), mode="constant")
    u8 = (values & np.int16(0xFF)).astype(np.uint16)
    words = u8[0::2] | (u8[1::2] << np.uint16(8))
    words.astype("<u2", copy=False).tofile(bin_out)


def creat_bin(data_in, output_name):
    if 'wt' in output_name:
        CHout, CHin, Ky, Kx = data_in.shape
        Tdata = parallel_weight    (data_in, CHout, CHin, Ky, Kx)
    elif 'bn' in output_name:
        CHout,_ = data_in.shape
        Tdata     = parallel_of_bn     (data_in, CHout)
    else:
        CHin,CHout,Hout,Wout = data_in.shape
        if CHin == 1:
            data_in = data_in.reshape((CHout,Hout,Wout))
        if CHout == 1:
            data_in = data_in.reshape((CHin,Hout,Wout))
    
        Tdata = parallel_of_feature(data_in, CHout,Hout,Wout)

        
    
    output_name = output_name.replace('txt', 'bin')
    output_name = os.path.join(os.path.dirname(output_name), output_name.split('/')[-1].replace('T32', 'T32_T32'))
    if 'wt.bin' in output_name:
        print(output_name, '   int8')
        INTarray_to_bin  (Tdata.flatten(), output_name) 
    else:
        print(output_name, '   fl')
        fp16array_to_bin (Tdata.flatten(), output_name)
