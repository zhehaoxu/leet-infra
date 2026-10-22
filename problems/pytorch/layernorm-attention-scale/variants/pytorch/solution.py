"""参考答案：QK-Norm 注意力 —— 先对 Q/K 每行做 RMSNorm 再点积除 sqrt(d)。"""

import math

import torch


def _rmsnorm(t: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    ms = t.pow(2).mean(-1, keepdim=True)
    return t * torch.rsqrt(ms + eps) * weight


def solve(q: torch.Tensor, k: torch.Tensor, weight_q: torch.Tensor, weight_k: torch.Tensor,
          eps: float = 1e-6) -> torch.Tensor:
    d = q.shape[-1]
    qn = _rmsnorm(q, weight_q, eps)
    kn = _rmsnorm(k, weight_k, eps)
    return torch.matmul(qn, kn.transpose(-2, -1)) / math.sqrt(d)
