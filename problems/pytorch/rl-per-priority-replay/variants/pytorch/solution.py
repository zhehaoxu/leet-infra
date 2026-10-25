"""参考答案：PER 优先级采样 + 重要性采样权重（全程 float64 累加，输出 float32）。"""

import torch


def solve(priorities, uniform_draws, alpha, beta):
    pri = priorities.reshape(-1).to(torch.float64).clamp(min=0.0)
    u = uniform_draws.reshape(-1).to(torch.float64)
    n = int(pri.numel())
    k = int(u.numel())
    a = float(alpha)
    b = float(beta)

    if n == 0:                        # 空池：无样本可采，下标 -1、权重 0
        return (torch.full((k,), -1, dtype=torch.long),
                torch.zeros(k, dtype=torch.float32),
                torch.zeros(0, dtype=torch.float32))

    powered = torch.pow(pri, a)          # 0**0 == 1，故 alpha=0 天然全 1
    total = float(powered.sum())
    if total == 0.0:                      # 全零优先级 + alpha>0：退化口径
        probs = torch.full((n,), 1.0 / n, dtype=torch.float64)
    else:
        probs = powered / total

    if k == 0:
        return (torch.zeros(0, dtype=torch.long),
                torch.zeros(0, dtype=torch.float32),
                probs.to(torch.float32))

    cdf = torch.cumsum(probs, dim=0)      # 向量化：对累积分布做二分查找
    idx = torch.searchsorted(cdf, u, right=True).clamp_(max=n - 1)

    raw = torch.pow(float(n) * probs[idx], -b)
    weights = raw / raw.max()
    return idx.to(torch.long), weights.to(torch.float32), probs.to(torch.float32)
