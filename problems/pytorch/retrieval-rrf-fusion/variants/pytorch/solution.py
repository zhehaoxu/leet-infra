"""参考答案：向量化 1/(k+r) 掩码求和，再按 (-score, idx) 稳定排序。"""

import torch


def solve(ranks, k):
    r = ranks.to(torch.float64)
    contrib = torch.where(r > 0, 1.0 / (k + r), torch.zeros_like(r))
    score = contrib.sum(dim=0)                       # (N,)
    return torch.argsort(-score, stable=True).to(torch.long)
