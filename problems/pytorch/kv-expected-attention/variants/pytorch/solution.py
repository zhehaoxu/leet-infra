import math

import torch
import torch.nn.functional as F


def solve(mu, cov, keys, values, n_sink, epsilon):
    """对齐 kvpress ExpectedAttentionPress.score（无 batch、kv_group=1、含 cov 与 vnorm）。"""
    H, L, D = keys.shape
    k = keys[:, n_sink:, :]                     # (H, L', D)
    v = values[:, n_sink:, :]
    # 一阶项：mu @ k / sqrt(D)
    s1 = (mu.unsqueeze(-2) @ k.transpose(-2, -1)).squeeze(-2) / math.sqrt(D)
    # 二阶项：k^T Σ k / (2D)
    s2 = torch.einsum('hni,hij,hnj->hn', k, cov, k) / D / 2
    scores = F.softmax(s1 + s2, dim=-1)
    scores = (scores + epsilon) * torch.linalg.vector_norm(v, dim=-1)
    m = scores.max().item() + 1.0
    return F.pad(scores, (n_sink, 0), value=m)
