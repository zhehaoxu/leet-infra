"""参考答案：只沿 xhat 方向做一次投影，没有回到零均值的那一项。

rms 是末轴上 x 的模长（差一个 1/H 因子），所以 d rms 只与 g·x 有关：
dX = g/rms − x · (Σ g·x)/(H·rms³)，整理成 (g − xhat·mean(g·xhat))/rms 与 LayerNorm
对照最清楚——少了 mean(g) 这一项，因为 RMSNorm 没有「减均值」这条依赖路径。
dGamma 依旧是把所有前导轴折叠后与 xhat 相乘求和。
"""

import torch


def solve(x, gamma, dY, eps=1e-6):
    if eps is None or float(eps) < 0:
        raise ValueError("eps 不得为负")
    if gamma.dim() != 1 or x.dim() < 1 or gamma.shape[0] != x.shape[-1]:
        raise ValueError("gamma 必须是与末轴等长的一维张量")
    if tuple(x.shape) != tuple(dY.shape):
        raise ValueError("dY 与 x 形状必须一致")

    xd = x.double()
    gd = gamma.double()
    dYd = dY.double()
    h = xd.shape[-1]

    ms = (xd * xd).mean(dim=-1, keepdim=True)
    rms = (ms + float(eps)).sqrt()
    xhat = xd / rms

    g = dYd * gd
    dX = (g - xhat * (g * xhat).mean(dim=-1, keepdim=True)) / rms

    dGamma = (dYd.reshape(-1, h) * xhat.reshape(-1, h)).sum(dim=0)
    return dX.to(x.dtype), dGamma.to(gamma.dtype)
