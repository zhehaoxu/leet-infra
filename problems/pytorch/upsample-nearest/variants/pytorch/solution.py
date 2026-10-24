"""参考答案：最近邻上采样（nearest-neighbor upsampling）。

每个输出像素直接复制其映射到的输入像素：out[i] = x[i // scale]（等价
F.interpolate(mode='nearest', scale_factor=s)，整数倍时即 floor(i/s) 索引）。
"""

import torch


def solve(x, scale):
    n, c, h, w = x.shape
    if isinstance(scale, int):
        sh = sw = scale
    else:
        sh, sw = scale
    oh, ow = h * sh, w * sw
    out = x.new_zeros((n, c, oh, ow))
    for i in range(oh):
        for j in range(ow):
            out[:, :, i, j] = x[:, :, i // sh, j // sw]
    return out
