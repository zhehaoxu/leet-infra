import torch


def solve(x, weight, bias, stride=1, padding=0):
    # TODO: 转置卷积。weight 形状 (Cin, Cout, kh, kw)（与 conv2d 相反）。
    # 输出尺寸 oh=(H-1)*stride-2*padding+kh。对每个输入元素把 weight[ci,:,ky,kx]
    # 累加到输出位置 [iy*stride-padding+ky, ix*stride-padding+kx]，越界丢弃；最后加 bias。
    raise NotImplementedError
