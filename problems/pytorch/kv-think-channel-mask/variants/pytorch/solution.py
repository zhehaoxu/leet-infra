import torch


def solve(queries, keys, compression_ratio):
    """对齐 kvpress ThinKPress.compress（kv_group=1、无 RoPE、置零不改 shape）。"""
    H, W, D = queries.shape
    L = keys.shape[1]
    q_norm = (queries.float() ** 2).mean(dim=1)  # (H, D)
    k_norm = (keys.float() ** 2).mean(dim=1)      # (H, D)
    key_scores = q_norm * k_norm
    n_pruned = int(D * compression_ratio)
    if n_pruned == 0:
        return keys.clone()
    idx = torch.topk(key_scores, n_pruned, dim=-1, largest=False).indices  # (H, n_pruned)
    idx_exp = idx.unsqueeze(1).expand(-1, L, -1)  # (H, L, n_pruned)
    return keys.scatter(-1, idx_exp, 0.0)
