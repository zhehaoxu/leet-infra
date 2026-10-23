"""参考答案：二维转置卷积 conv_transpose2d（散射累加式教学实现）。

weight 形状 (Cin, Cout, kh, kw)（注意与 conv2d 相反）。输出尺寸
oh=(H-1)*stride-2*padding+kh。对每个输入元素，把 weight[ci,:,ky,kx] 按
输出位置 [iy*stride-padding+ky, ix*stride-padding+kx] 累加进去（越界丢弃），
等价于 conv2d 对输入的梯度。
"""

import torch


def solve(x, weight, bias, stride=1, padding=0):
    n, cin, h, w = x.shape
    cout, kh, kw = weight.shape[1], weight.shape[2], weight.shape[3]
    oh = (h - 1) * stride - 2 * padding + kh
    ow = (w - 1) * stride - 2 * padding + kw
    out = x.new_zeros((n, cout, oh, ow))
    for ci in range(cin):
        for iy in range(h):
            for ix in range(w):
                oy0 = iy * stride - padding
                ox0 = ix * stride - padding
                val = x[:, ci, iy, ix].view(-1, 1)              # (n, 1)
                for ky in range(kh):
                    oy = oy0 + ky
                    if oy < 0 or oy >= oh:
                        continue
                    for kx in range(kw):
                        ox = ox0 + kx
                        if ox < 0 or ox >= ow:
                            continue
                        out[:, :, oy, ox] += val * weight[ci, :, ky, kx].view(1, -1)
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)
    return out
