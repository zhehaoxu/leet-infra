"""参考答案：三维卷积 conv3d（朴素教学实现，scatter 式跨步切片累加）。

先在三个空间维各向同性补零，再对每个输出通道 o、输入通道 ci、核内偏移 (di,hi,wi)，
把标量权重乘到输入的跨步切片上整片累加到 out[:, o]，最后加 bias。核不翻转，与 F.conv3d 一致。
"""

import torch
import torch.nn.functional as F


def solve(x, weight, bias, stride=1, padding=0):
    n, cin, d, h, w = x.shape
    cout, _, kd, kh, kw = weight.shape
    xp = F.pad(x, (padding, padding, padding, padding, padding, padding))
    od = (xp.shape[2] - kd) // stride + 1
    oh = (xp.shape[3] - kh) // stride + 1
    ow = (xp.shape[4] - kw) // stride + 1
    out = x.new_zeros((n, cout, od, oh, ow))
    for o in range(cout):
        for ci in range(cin):
            for di in range(kd):
                for hi in range(kh):
                    for wi in range(kw):
                        out[:, o] += weight[o, ci, di, hi, wi] * xp[
                            :, ci,
                            di:di + od * stride:stride,
                            hi:hi + oh * stride:stride,
                            wi:wi + ow * stride:stride,
                        ]
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1, 1)
    return out
