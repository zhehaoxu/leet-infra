"""参考答案：把每个激活的导数写成 x 的闭式，再与 dY 相乘。

要点全在零点约定与「导数用前向输出表示」两件事上：
- relu 在 0 处 torch 给 0（不是 0.5），gelu/silu 因为在 0 处光滑所以是 0.5/0.5；
- sigmoid/tanh 用前向输出 s、t 表达导数 s(1−s)、1−t² 最省算力，也最不容易溢出；
- gelu 的精确式含 erf，写成 tanh 近似版会在 |x|≈1 处与 autograd 出现 1e-3 量级偏差。
"""

import math

import torch


def _sigmoid(t):
    return torch.sigmoid(t)


def solve(x, dY, kind, negative_slope=0.01):
    if tuple(x.shape) != tuple(dY.shape):
        raise ValueError("x 与 dY 形状必须一致")
    if not isinstance(kind, str):
        raise ValueError("kind 必须是字符串")
    k = kind.lower()
    if k not in ('relu', 'leaky_relu', 'elu', 'sigmoid', 'tanh', 'gelu', 'silu', 'softplus'):
        raise ValueError(f"不支持的 kind: {kind}")

    xd = x.double()
    gd = dY.double()
    ns = float(negative_slope)

    if k == 'relu':
        dphi = (xd > 0).to(torch.float64)
    elif k == 'leaky_relu':
        dphi = torch.where(xd > 0, torch.ones_like(xd), torch.full_like(xd, ns))
    elif k == 'elu':
        dphi = torch.where(xd > 0, torch.ones_like(xd), xd.exp())
    elif k == 'sigmoid':
        s = _sigmoid(xd)
        dphi = s * (1.0 - s)
    elif k == 'tanh':
        dphi = 1.0 - xd.tanh().pow(2)
    elif k == 'softplus':
        dphi = _sigmoid(xd)
    elif k == 'silu':
        s = _sigmoid(xd)
        dphi = s + xd * s * (1.0 - s)
    else:  # gelu（精确式）
        cdf = 0.5 * (1.0 + torch.erf(xd / math.sqrt(2.0)))
        pdf = torch.exp(-0.5 * xd * xd) / math.sqrt(2.0 * math.pi)
        dphi = cdf + xd * pdf

    return (gd * dphi).to(x.dtype)
