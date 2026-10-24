import torch


FLT_MAX = torch.finfo(torch.float32).max


def solve(scores, compression_ratio, alpha_safeguard):
    """对齐 kvpress AdaKVPress.compress（无 batch）。"""
    s = scores.float().clone()
    H, L = s.shape
    n_kept = int(L * (1 - compression_ratio))
    n_safe = int(n_kept * alpha_safeguard)
    if alpha_safeguard > 0:
        n_safe = max(1, n_safe) if n_kept > 0 else 0
    if n_safe > 0:
        top_idx = torch.topk(s, n_safe, dim=-1).indices
        s.scatter_(-1, top_idx, FLT_MAX)
    n_pruned = H * (L - n_kept)
    if n_pruned <= 0:
        return torch.ones(H, L, dtype=torch.bool)
    flat = s.reshape(-1)
    prune_idx = torch.topk(-flat, n_pruned, dim=0).indices
    mask = torch.ones(H * L, dtype=torch.bool)
    mask[prune_idx] = False
    return mask.view(H, L)
