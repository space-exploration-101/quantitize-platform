import _pin_stdlib_platform  # noqa: F401

import torch
import struct
import numpy as np
import pickle
import os
import random
import numba as nb
from numba import jit
import binascii


def parallel_of_feature(input, CH, H, W):
    slice_of_CH = (CH + Tout - 1) // Tout
    output = np.zeros((slice_of_CH, H, W, Tout))
    for h in range(H):
        for w in range(W):
            for ch in range(slice_of_CH):
                for t in range(Tout):
                    if ch * Tout + t < CH:
                        tp = input[ch * Tout + t][h][w]
                    else:
                        tp = 0
                    output[ch][h][w][t] = tp
    return output


def demap_of_feature(input, CH, H, W):
    slice_of_CH = (CH + Tout - 1) // Tout
    input = input.reshape(slice_of_CH, H, W, Tout)
    output = np.zeros((CH, H, W))
    for h in range(H):
        for w in range(W):
            for ch in range(slice_of_CH):
                for t in range(Tout):
                    output[ch * Tout + t][h][w] = input[ch][h][w][t]
    return output


def demap_of_bn(bn_input, CH):
    bn_output = bn_input.reshape(CH, 2)
    return bn_output



def parallel_weight(weight_input, CHout, CHin, Ky, Kx):
    slice_of_CHin = (CHin + Tin - 1) // Tin
    slice_of_CHout = (CHout + Tout - 1) // Tout
    weight_reorg = np.zeros((slice_of_CHout, slice_of_CHin, Ky, Kx, Tout, Tin))
    for chout in range(slice_of_CHout):
        for chin in range(slice_of_CHin):
            for ky in range(Ky):
                for kx in range(Kx):
                    for tout in range(Tout):
                        for tin in range(Tin):
                            if chout * Tout + tout < CHout and chin * Tin + tin < CHin:
                                tp1 = weight_input[chout * Tout + tout][chin * Tin + tin][ky][kx]
                                # print(chout, chin, ky, kx, tout, ll, tout)
                            else:
                                tp1 = 0
                            weight_reorg[chout][chin][ky][kx][tout][tin] = tp1
    return weight_reorg


def demap_of_weight(input, CHout, CHin, Ky, Kx):
    slice_of_CHin = (CHin + Tin - 1) // Tin
    slice_of_CHout = (CHout + Tout - 1) // Tout
    input = input.reshape(slice_of_CHout, slice_of_CHin, Ky, Kx, Tout, Tin)
    output = np.zeros((CHout, CHin, Ky, Kx))
    for chout in range(slice_of_CHout):
        for chin in range(slice_of_CHin):
            for ky in range(Ky):
                for kx in range(Kx):
                    for tout in range(Tout):
                        for tin in range(Tin):
                            output[chout * Tout + tout][chin * Tin + tin][ky][kx] = input[chout][chin][ky][kx][tout][
                                tin]
    return output


def read_fp_bin_file_with_numpy(file_path, dtype):
    out_array = np.fromfile(file_path, dtype=dtype)
    # print(out_array)
    return out_array


def read_int_bin_file_with_numpy(file_path, dw):
    in_array = np.fromfile(file_path, dtype=np.short)
    count = len(in_array)
    if (dw == 16):
        out_array = np.zeros((count))
        for i in range(count):
            out_array[i] = in_array[i]
    elif (dw == 8):
        out_array = np.zeros((count * 2))
        for i in range(count):
            out_array[2 * i + 0] = in_array[i] % 256
            out_array[2 * i + 1] = in_array[i] // 256
    elif (dw == 4):
        out_array = np.zeros((count * 4))
        for i in range(count):
            out_array[4 * i + 0] = in_array[i] % 16
            out_array[4 * i + 1] = (in_array[i] // 16) % 16
            out_array[4 * i + 2] = (in_array[i] // 256) % 16
            out_array[4 * i + 3] = (in_array[i] // 4096) % 16
    else:
        out_array = np.zeros((count))

    for i in range(len(out_array)):
        if (out_array[i] > 2 ** (dw - 1) - 1):
            out_array[i] = out_array[i] - (2 ** dw)
    return (out_array)


def INTarray_to_bin(dat_in, bin_out):
    count = len(dat_in)
    # print(count)
    mod = count % 2
    if (mod):
        dat_out_0 = np.zeros((count + 2 - mod) // 2)
    else:
        dat_out_0 = np.zeros(count // 2)

    if (mod):
        print("txt行数不是2的倍数，补充", (2 - mod), "行0")
        for i in range(2 - mod):
            dat_in = np.append(dat_in, 0)
        # print(dat_in)

    for i in range(dat_out_0.size):
        a1 = int(dat_in[2 * i])
        if (a1 < 0):
            a1 = 256 + a1
        a1 = a1

        a2 = int(dat_in[2 * i + 1])
        if (a2 < 0):
            a2 = 256 + a2
        a2 = a2 * 256

        b = a1 + a2
        if ((b <= 32767) & (b >= -32768)):
            b = b
        else:
            b = b - 65536
        dat_out_0[i] = b

    dat_out_1 = open(bin_out, 'wb')
    size = dat_out_0.size
    for i in range(size):
        a = int(dat_out_0[i])
        b = struct.pack('h', a)
        dat_out_1.write(b)
    dat_out_1.close()


def fp16array_to_bin(input_array, output_file):
    float_array = np.array(input_array, dtype=np.float32)  # 转换为numpy数组
    fp16_array = float_array.astype(np.float16)  # 转换为FP16格式
    fp16_array.tofile(output_file)  # 保存为二进制文件


def INT_txt_to_bin(txt_in, bin_out,dw):
    if(dw==4):
        count = len(open(txt_in, 'r').readlines())
        #print(count)
        dat_in = np.loadtxt(txt_in)
        #print(dat_in)
        mod = count%4
        if (mod):
            dat_out_0 = np.zeros((count + 4 - mod) // 4)
        else:
            dat_out_0 = np.zeros(count // 4)
        print("size of 16bit:", dat_out_0.size)
        if(mod):
            print("txt行数不是4的倍数，补充", (4-mod),"行0")
            for i in range(4-mod):
                dat_in = np.append(dat_in,0)
            #print(dat_in)

        for i in range(dat_out_0.size):
            a1 = int(dat_in[4 * i])
            if (a1 < 0):
                a1 = 16 + a1
            a1 = a1

            a2 = int(dat_in[4 * i + 1])
            if (a2 < 0):
                a2 = 16 + a2
            a2 = a2 * 16

            a3 = int(dat_in[4 * i + 2])
            if (a3 < 0):
                a3 = 16 + a3
            a3 = a3 * (16 ** 2)

            a4 = int(dat_in[4 * i + 3])
            if (a4 < 0):
                a4 = 16 + a4
            a4 = a4 * (16 ** 3)

            b = a1 + a2 + a3 + a4
            if ((b <= 32767 )& (b >= -32768)):
                b=b
            else:
                b = b - 65536
            dat_out_0[i] = b

        dat_out_1 = open(bin_out, 'wb')
        size = dat_out_0.size
        for i in range(size):
            a = int(dat_out_0[i])
            b = struct.pack('h', a)
            dat_out_1.write(b)
        dat_out_1.close()


    elif(dw==8):
        count = len(open(txt_in, 'r').readlines())
        #print(count)
        dat_in = np.loadtxt(txt_in)
        #print(dat_in)
        mod = count%2
        if (mod):
            dat_out_0 = np.zeros((count + 2 - mod) // 2)
        else:
            dat_out_0 = np.zeros(count // 2)
        print("size of 16bit:", dat_out_0.size)
        if(mod):
            print("txt行数不是2的倍数，补充", (2-mod),"行0")
            for i in range(2-mod):
                dat_in = np.append(dat_in,0)
            #print(dat_in)

        for i in range(dat_out_0.size):
            a1 = int(dat_in[2 * i])
            if (a1 < 0):
                a1 = 256 + a1
            a1 = a1

            a2 = int(dat_in[2 * i + 1])
            if (a2 < 0):
                a2 = 256 + a2
            a2 = a2 * 256


            b = a1 + a2
            if ((b <= 32767 )& (b >= -32768)):
                b=b
            else:
                b = b - 65536
            dat_out_0[i] = b

        dat_out_1 = open(bin_out, 'wb')
        size = dat_out_0.size
        for i in range(size):
            a = int(dat_out_0[i])
            b = struct.pack('h', a)
            dat_out_1.write(b)
        dat_out_1.close()

    elif(dw==16):
        count = len(open(txt_in, 'r').readlines())
        dat_in = np.loadtxt(txt_in)
        print("size of 16bit:", 2*count)

        dat_out = open(bin_out, 'wb')
        for i in range(count):
            a = int(dat_in[i])
            b = struct.pack('h', a)
            dat_out.write(b)
        dat_out.close()

    else:
        print("wt width should be 2/4/8/16 bit")



def fp16_txt_to_bin(input_file, output_file):
    with open(input_file, 'r') as f:
        data = f.readlines()
    float_data = [float(line.strip()) for line in data]  # 将字符串转换为浮点数
    float_array = np.array(float_data, dtype=np.float32) # 转换为numpy数组
    fp16_array = float_array.astype(np.float16)          # 转换为FP16格式
    fp16_array.tofile(output_file)                       # 保存为二进制文件


if __name__ == "__main__":
    Win = 80
    Hin = 80
    CHin = 320
    CHout = 2
    Ky = 1
    Kx = 1
    Sy = 1
    Sx = 1
    Py = 0
    Px = 0
    CONV_RELU = 0
    BN_RELU = 0
    SILU_EN = 0

    ###############  software defines finish #######################

    base_Tin = 32
    Tout = 32
    Tin = base_Tin
    Tb = 1
    ###############  software defines finish #######################

    Hout = (Hin - Ky + 2 * Py) // Sy + 1
    Wout = (Win - Kx + 2 * Px) // Sx + 1
    CHout_Padding_with_Tout = (CHout + Tout - 1)//Tout * Tout
    CHin_Padding_with_Tin = (CHin + Tin - 1)//Tin * Tin

    T_fp_bn      = read_fp_bin_file_with_numpy( "./BN/T32_T32_L02_s00_conv_bn.bin", dtype=np.float16)
    fp_bn = demap_of_bn(T_fp_bn, 160)

    new_fp_bn = np.zeros((192,2))
    for i in range(192):
        for j in range (2):
            if (i < 80):
                new_fp_bn[i, j] = fp_bn[i, j]
            elif (i < 96):
                new_fp_bn[i, j] = 0
            elif (i < 176):
                new_fp_bn[i, j] = fp_bn[i - 16, j]
            else:
                new_fp_bn[i, j] = 0

    T_new_fp_bn = demap_of_bn(new_fp_bn, 192)
    np.savetxt("T32_T32_L02_s00_conv_bn.txt", T_new_fp_bn.flatten(), fmt='%f')
    fp16_txt_to_bin ("T32_T32_L02_s00_conv_bn.txt", "T32_T32_L02_s00_conv_bn.bin")
