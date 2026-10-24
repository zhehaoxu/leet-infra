import torch


def solve(k_len, n_sink, compression_ratio):
    """对齐 kvpress StreamingLLMPress：sink 与尾窗=1，中段 n_pruned 位置=0。"""
    n_pruned = k_len - int(k_len * (1 - compression_ratio))
    scores = torch.ones(k_len, dtype=torch.float32)
    scores[n_sink:n_sink + n_pruned] = 0.0
    return scores
