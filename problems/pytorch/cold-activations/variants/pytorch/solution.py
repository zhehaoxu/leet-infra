"""参考答案：cold-activations 四种逐元素激活的分派实现（纯 torch、无随机数）。

logsigmoid 用 -softplus(-x) 稳定式；celu 用 float64 中间精度 + expm1，
保证大 alpha 时与公式 max(0,x)+min(0,alpha·(exp(x/alpha)-1)) 数学恒等且数值稳定。
"""

import torch


def _softplus_stable(z):
    """与 F.softplus(beta=1, threshold=20) 同构：z > 20 走线性段，其余 log1p(exp(z))。"""
    return torch.where(z > 20.0, z, torch.log1p(torch.exp(z)))


def solve(x, mode="softsign", min_value=-1.0, max_value=1.0, alpha=1.0):
    if mode == "softsign":
        return x / (1.0 + x.abs())
    if mode == "hardtanh":
        return x.clamp(min=min_value, max=max_value)
    if mode == "celu":
        # 在 float64 下计算：exp(x/alpha)-1 在 x/alpha 极小（大 alpha）时灾难性抵消，
        # 用 expm1 且抬精度后与 F.celu（内部同样走高精度 opmath）保持一致。
        xd = x.to(torch.float64)
        pos = xd.clamp_min(0.0)
        neg = alpha * torch.expm1(xd.clamp_max(0.0) / alpha)
        return (pos + neg).to(x.dtype)
    if mode == "logsigmoid":
        return -_softplus_stable(-x)
    raise ValueError(f"unknown mode: {mode!r}")
