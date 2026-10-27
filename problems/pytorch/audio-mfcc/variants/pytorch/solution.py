import math

import torch


def solve(logmel, n_mfcc, lifter=0, cmvn=False):
    if not isinstance(logmel, torch.Tensor) or logmel.dim() != 2:
        raise ValueError("logmel 必须是二维张量")
    n_mel = int(logmel.size(1))
    if n_mfcc < 1 or n_mfcc > n_mel:
        raise ValueError("n_mfcc 必须落在 [1, n_mel]")
    if lifter < 0:
        raise ValueError("lifter 不能为负")

    x = logmel.to(torch.float64)
    k = torch.arange(n_mfcc, dtype=torch.float64)[:, None]
    n = torch.arange(n_mel, dtype=torch.float64)[None, :]
    basis = torch.cos(math.pi * k * (n + 0.5) / n_mel) * math.sqrt(2.0 / n_mel)
    basis[0] = basis[0] / math.sqrt(2.0)  # k=0 的 s_0 = sqrt(1/M)

    ceps = x @ basis.T
    if lifter > 0:
        idx = torch.arange(n_mfcc, dtype=torch.float64)
        ceps = ceps * (1.0 + 0.5 * lifter * torch.sin(math.pi * idx / lifter))[None, :]

    if cmvn:
        mean = ceps.mean(dim=0, keepdim=True)
        std = ceps.std(dim=0, unbiased=False, keepdim=True)
        ceps = (ceps - mean) / (std + 1e-8)

    return ceps.to(torch.float32)
