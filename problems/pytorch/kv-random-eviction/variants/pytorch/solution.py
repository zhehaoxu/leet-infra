"""参考答案：Random eviction 基线（显式循环版，判题用）。"""

import torch


def solve(draws, attn_weights, n_sink, window_size, compression_ratio):
    L = draws.shape[0]
    n_kept = max(1, int(L * (1 - compression_ratio)))
    # 保底区：sink ∪ recent 窗口，恒保留
    forced = []
    for j in range(L):
        if j < n_sink or j >= L - window_size:
            forced.append(j)
    forced_set = set(forced)
    middle = [j for j in range(L) if j not in forced_set]
    n_mid = n_kept - len(forced_set)
    if n_mid < 0:
        n_mid = 0
    if n_mid > len(middle):
        n_mid = len(middle)
    chosen = sorted(middle, key=lambda j: (-float(draws[j]), j))[:n_mid]
    kept = sorted(forced_set | set(chosen))
    total = 0.0
    for j in range(L):
        total += float(attn_weights[j])
    if total > 0.0:
        hit = 0.0
        for j in kept:
            hit += float(attn_weights[j])
        coverage = hit / total
    else:
        coverage = len(kept) / L
    return torch.tensor(kept, dtype=torch.long), coverage
