"""参考答案：Welford 单遍更新 (n, mean, M2)，最后一步才做除法。

教学核心是「为什么不能用 sum of squares 减 mean 平方」。对 [1e9+1, 1e9+2, 1e9+3, 1e9+4]，
真实样本方差是 5/3 ≈ 1.6666666666666667；而 sum(x^2) 已经在 1e18 量级，double 只有约
15~16 位十进制有效数字，个位数被舍掉，E[x^2] - mean^2 直接归零。Welford 把偏差 d 在
每一小步就地缩小到 1 量级，M2 累的是 d*(v-mean_new)，量级始终贴着真实离散度，不受偏置影响。

d 用的是「旧均值」，M2 里第二个因子用的是「更新后的均值」，两者混用正是它比
M2 += d*d 精确的原因，写反或写成同一个均值是最常见的错。
"""

import math

import torch


FLOAT_DTYPES = (torch.float32, torch.float64)


def _validate(chunks, unbiased):
    if not isinstance(chunks, (list, tuple)):
        raise ValueError("chunks 必须是 list 或 tuple")
    if not isinstance(unbiased, bool):
        raise ValueError("unbiased 必须是 bool")
    for c in chunks:
        if not isinstance(c, torch.Tensor):
            raise ValueError("chunks 的元素必须是 torch.Tensor")
        if c.dtype not in FLOAT_DTYPES:
            raise ValueError("chunks 的元素必须是 float32 或 float64 张量")


def solve(chunks, unbiased=True):
    _validate(chunks, unbiased)
    n = 0
    mean = 0.0
    m2 = 0.0
    for c in chunks:
        for v in c.detach().to(torch.float64).reshape(-1).tolist():
            if not math.isfinite(v):
                return (float('nan'), float('nan'))
            n += 1
            d = v - mean
            mean += d / n
            m2 += d * (v - mean)
    if n == 0:
        return (float('nan'), float('nan'))
    if n == 1:
        return (mean, float('nan') if unbiased else 0.0)
    var = m2 / (n - 1) if unbiased else m2 / n
    return (mean, var)
