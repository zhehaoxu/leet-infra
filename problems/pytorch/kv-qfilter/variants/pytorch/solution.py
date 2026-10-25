"""参考答案：Q-Filter 学习滤波器负点积打分（向量化版，判题用）。"""

import torch


def solve(q_filters, keys, compression_ratio):
    N, H, L, D = keys.shape
    # scores = -(q_filter * keys).sum(-1)，逐 (layer, kv_head)
    scores = -(q_filters[:, :, None, :] * keys).sum(dim=-1)   # (N, H, L)
    n_kept = max(1, int(L * (1 - compression_ratio)))
    order = torch.sort(-scores, dim=-1, stable=True).indices[:, :, :n_kept]
    kept = torch.sort(order, dim=-1).values
    return scores, kept
