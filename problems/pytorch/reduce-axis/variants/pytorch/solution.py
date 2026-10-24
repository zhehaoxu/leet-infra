"""参考答案：沿指定轴求和归约（不使用 torch.sum）。

把目标轴沿切片方向逐份累加：acc = x[0] + x[1] + ... + x[n-1]（在 axis 上），
keepdim=True 时再 unsqueeze 回原轴，形状与 torch.sum(x, axis, keepdim=...) 一致。
"""

import torch


def solve(x, axis, keepdim=False):
    if axis < 0:
        axis = x.dim() + axis
    n = x.shape[axis]

    def sl(i):
        s = [slice(None)] * x.dim()
        s[axis] = i
        return tuple(s)

    acc = x[sl(0)].clone()
    for i in range(1, n):
        acc = acc + x[sl(i)]
    if keepdim:
        acc = acc.unsqueeze(axis)
    return acc
