"""参考答案：一维卷积 conv1d（朴素教学实现，手写滑动窗口 + 通道求和）。

先在长度维两侧补零，再对每个输出通道 o、每个输出位置 t，把所有输入通道 ci 与
核 window 逐元素相乘求和，最后加 bias。核不翻转（互相关约定），与 F.conv1d 一致。
"""

import torch
import torch.nn.functional as F


def solve(x, weight, bias, stride=1, padding=0):
    n, cin, l = x.shape
    cout, _, kk = weight.shape
    xp = F.pad(x, (padding, padding))
    L = (xp.shape[2] - kk) // stride + 1
    out = x.new_zeros((n, cout, L))
    for o in range(cout):
        for ci in range(cin):
            for t in range(L):
                out[:, o, t] += (xp[:, ci, t * stride:t * stride + kk] * weight[o, ci]).sum(-1)
    if bias is not None:
        out = out + bias.view(1, -1, 1)
    return out
