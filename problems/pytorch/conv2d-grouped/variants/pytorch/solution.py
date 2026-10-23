"""参考答案：分组卷积 conv2d with groups（朴素教学实现，沿用 conv2d-naive 的逐输出通道相关）。

weight 形状 (Cout, Cin//groups, kh, kw)：输入按通道切成 groups 段，每段只在自己的
输入通道上卷积、产出 Cout//groups 个输出通道，组与组之间不相连。
"""

import torch


def solve(x, weight, bias, groups):
    n, cin, h, w = x.shape
    cout, cin_k, kh, kw = weight.shape
    in_per_g = cin // groups
    out_per_g = cout // groups
    assert cin_k == in_per_g, "权重输入通道数须为 Cin//groups"
    oh = h - kh + 1
    ow = w - kw + 1
    out = x.new_zeros((n, cout, oh, ow))
    for g in range(groups):
        ci0 = g * in_per_g
        co0 = g * out_per_g
        xg = x[:, ci0:ci0 + in_per_g]
        wg = weight[co0:co0 + out_per_g]                          # (out_per_g, in_per_g, kh, kw)
        for i in range(oh):
            for j in range(ow):
                region = xg[:, :, i:i + kh, j:j + kw]
                for o in range(out_per_g):
                    out[:, co0 + o, i, j] = (region * wg[o]).sum(dim=(1, 2, 3))
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)
    return out
