"""参考答案：沿指定轴的累积和（前缀和 scan），不使用 torch.cumsum。

out[i] = x[0] + x[1] + ... + x[i]（沿 axis）。用「上一份累计 + 当前元素」递推：
out[i] = out[i-1] + x[i]，out[0] = x[0]。
"""

import torch


def solve(x, axis):
    if axis < 0:
        axis = x.dim() + axis
    n = x.shape[axis]

    def sl(i):
        s = [slice(None)] * x.dim()
        s[axis] = i
        return tuple(s)

    out = x.clone()
    for i in range(1, n):
        out[sl(i)] = out[sl(i - 1)] + x[sl(i)]
    return out
