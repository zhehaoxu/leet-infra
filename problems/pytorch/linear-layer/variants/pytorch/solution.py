"""参考答案：全连接（线性）层 y = x·Wᵀ + b。

weight 采用 PyTorch nn.Linear 的约定 (out_features, in_features)：对最后一维做
「输入通道 × 权重」的乘加，逐输出单元求和。这里用切片累加显式实现矩阵乘，
不调用 torch.matmul/@，以突出「每个输出 = 输入与对应权重行的点积」。
"""

import torch


def solve(x, weight, bias=None):
    # x: (..., in), weight: (out, in)
    in_feat = weight.shape[1]
    out = x.new_zeros(x.shape[:-1] + (weight.shape[0],))
    for o in range(weight.shape[0]):
        acc = x.new_zeros(x.shape[:-1])
        for i in range(in_feat):
            sl = [slice(None)] * x.dim()
            sl[-1] = i
            acc = acc + x[tuple(sl)] * weight[o, i]
        out[..., o] = acc
    if bias is not None:
        out = out + bias
    return out
