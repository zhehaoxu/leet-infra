"""参考答案：块尺度用 2 的幂（frexp 精确定指数），元素量化用「按倍频取步长 + RNE 取整」。

尺度必须是最小的 2 的幂：这样乘除都是指数的平移，不引入新的舍入误差，反量化也精确。
用 m / 57344 这种「非 2 的幂」尺度会让每次乘除都多一道舍入，且块内最大值会被压到刚好贴边。
定指数走 frexp：x = f * 2^E（f ∈ [0.5,1)），那么 ceil(log2 x) 在 f == 0.5 时是 E-1，否则是 E——
完全避开 log2 的浮点误差。

元素量化不用查表：把幅值写成 a = k * 2^(b-2)（b 是 a 所在倍频），k 就落在 [4,8) 上，
一次 torch.round（本身就是 ties-to-even）就把「就近 + 取偶」做完；次正规区把指数夹到 -16
就退化成固定步长 2^-16 的均匀网格，正好和正规区在 2^-14 处无缝接上。
"""

import math

import torch

MAX_E5M2 = 57344.0
SUB_STEP_EXP = -16


def _scale_exp(m):
    """最小的整数 e 使 m <= MAX_E5M2 * 2^e（m > 0）"""
    if m == 0.0:
        return 0
    f, e = math.frexp(m / MAX_E5M2)
    exp = e - 1 if f == 0.5 else e
    return max(-127, min(127, exp))


def _snap(t):
    """把 float64 张量 t 舍到最近的 E5M2 格点（ties-to-even），保留符号"""
    a = t.abs()
    mant, expo = torch.frexp(a)          # a = mant * 2^expo, mant ∈ [0.5, 1)
    binade = expo - 1
    step_exp = torch.clamp(binade - 2, min=SUB_STEP_EXP)
    step = torch.ldexp(torch.ones_like(a), step_exp)
    q = torch.round(a / step) * step
    q = torch.where(a >= MAX_E5M2, torch.full_like(q, MAX_E5M2), q)
    return torch.where(t < 0, -q, q)


def solve(x, block=32):
    if not isinstance(x, torch.Tensor):
        raise ValueError("x 必须是 torch.Tensor")
    if x.dtype not in (torch.float32, torch.float64):
        raise ValueError("x 必须是 float32 或 float64 张量")
    if isinstance(block, bool) or not isinstance(block, int) or block < 1:
        raise ValueError("block 必须是 >= 1 的整数")

    src = x.detach().to(torch.float64)
    flat = src.reshape(-1)
    out = torch.empty_like(flat)
    n = flat.numel()
    for s in range(0, n, block):
        seg = flat[s:s + block]
        m = float(seg.abs().max())
        if not math.isfinite(m):
            # 块内有 nan/inf：absmax 无意义，整块判 NaN（避免 frexp(inf) 走偏）
            out[s:s + block] = float('nan')
            continue
        exp = _scale_exp(m)
        q = _snap(seg / (2.0 ** exp))
        out[s:s + block] = q * (2.0 ** exp)
    return out.reshape(src.shape).to(x.dtype)
