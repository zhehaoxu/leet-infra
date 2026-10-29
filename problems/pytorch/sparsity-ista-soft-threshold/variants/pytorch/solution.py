"""参考答案：ISTA 一步（梯度步 + L1 软阈值近端）。"""

import torch


def solve(x, W, y, lam, eta):
    r = W @ x - y
    g = W.transpose(0, 1) @ r
    z = x - eta * g
    x_new = torch.sign(z) * torch.clamp(z.abs() - lam, min=0.0)
    support = torch.nonzero(x_new.abs() > 0).flatten().to(torch.long)
    return x_new, support
