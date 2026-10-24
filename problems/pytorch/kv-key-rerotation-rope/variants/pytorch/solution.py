import torch


def _rotate_half(x):
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2:]
    return torch.cat([-x2, x1], dim=-1)


def solve(keys, indices, inv_freq):
    """KeyRerotation：gather → 按 (new_pos - old_pos)*inv_freq 重旋转。"""
    H, L, D = keys.shape
    n_kept = indices.shape[1]
    gidx = indices.unsqueeze(-1).expand(-1, -1, D)
    k = keys.gather(1, gidx)                                  # (H, n_kept, D)
    new_pos = torch.arange(n_kept).unsqueeze(0).expand(H, n_kept)
    delta = (new_pos - indices).to(keys.dtype)                # (H, n_kept)
    freqs = delta.unsqueeze(-1) * inv_freq.to(keys.dtype)     # (H, n_kept, D/2)
    emb = torch.cat([freqs, freqs], dim=-1)                   # (H, n_kept, D)
    return k * torch.cos(emb) + _rotate_half(k) * torch.sin(emb)
