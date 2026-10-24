import torch


def solve(attn, num_kv_heads):
    """对齐 kvpress ObservedAttentionPress.score（无 batch）。"""
    # attn (Hq, Q, K)
    scores = attn.sum(dim=1)                    # (Hq, K) — 对 query 维求和
    L = attn.shape[1]
    n_in_sum = torch.arange(L, 0, -1, dtype=attn.dtype)  # K=j 处 L-j
    scores = scores / n_in_sum
    groups = attn.shape[0] // num_kv_heads
    scores = scores.view(num_kv_heads, groups, L).mean(dim=1)
    return scores
