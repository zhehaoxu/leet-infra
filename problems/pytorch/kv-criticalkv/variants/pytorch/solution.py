"""参考答案：CriticalKV 两阶段保留（显式 Python 循环版，判题用）。"""

import torch


def solve(scores, wov_norm, compression_ratio, first_stage_ratio, epsilon):
    H, L = scores.shape
    finfo_max = torch.finfo(scores.dtype).max
    n_kept = max(1, int(L * (1 - compression_ratio)))
    budget1 = int((1 - compression_ratio) * L * first_stage_ratio)
    if first_stage_ratio > 0:
        budget1 = max(1, budget1)
    budget1 = min(budget1, L)
    kept_rows = []
    final = torch.empty(H, L, dtype=scores.dtype)
    for h in range(H):
        s = [float(scores[h, j]) for j in range(L)]
        n = [float(wov_norm[h, j]) for j in range(L)]
        # Stage 1：原始分 top-budget1（平局取小下标）
        stage1 = sorted(range(L), key=lambda j: (-s[j], j))[:budget1]
        # Stage 2：(score + eps) * ||Wo v||_1
        f = [(s[j] + epsilon) * n[j] for j in range(L)]
        for j in stage1:
            f[j] = finfo_max
        top2 = sorted(range(L), key=lambda j: (-f[j], j))[:n_kept]
        top2.sort()
        kept_rows.append(top2)
        for j in range(L):
            final[h, j] = f[j]
    return torch.tensor(kept_rows, dtype=torch.long), final
