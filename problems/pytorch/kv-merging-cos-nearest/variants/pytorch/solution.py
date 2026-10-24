import torch


_EPS = 1e-6


def solve(keys, values, scores, n_kept):
    """对齐 kvpress MergingPress.merge + compress（threshold=0、fraction=1、无 batch）。"""
    H, L, D = keys.shape
    n_evict = L - n_kept
    if n_evict == 0 or n_kept == 0:
        return values.clone().float()
    indices = torch.topk(scores.float(), n_kept, dim=-1).indices  # (H, n_kept) 降序

    evict_mask = torch.ones(H, L, dtype=torch.bool)
    evict_mask.scatter_(1, indices, False)
    # nonzero 按 flatten 顺序 → 每 h 的原位置升序
    evict_idx = evict_mask.nonzero(as_tuple=False)[:, 1].reshape(H, n_evict)

    keep_idx_e = indices.unsqueeze(-1).expand(-1, -1, D)
    evict_idx_e = evict_idx.unsqueeze(-1).expand(-1, -1, D)

    kk = keys.gather(1, keep_idx_e).float()
    ek = keys.gather(1, evict_idx_e).float()
    kv_ = values.gather(1, keep_idx_e).float()
    ev_ = values.gather(1, evict_idx_e).float()

    kk = kk / kk.norm(dim=-1, keepdim=True).clamp(min=_EPS)
    ek = ek / ek.norm(dim=-1, keepdim=True).clamp(min=_EPS)

    sim = ek @ kk.transpose(-2, -1)                            # (H, n_ev, n_kept)
    max_sim, target = sim.max(dim=-1)                          # (H, n_ev)

    weights = max_sim.clamp(min=0)                             # threshold=0
    tn = kv_.norm(dim=-1).gather(1, target)                    # (H, n_ev)
    en = ev_.norm(dim=-1)
    weights = weights * en / (en + tn + _EPS)

    tgt_e = target.unsqueeze(-1).expand(-1, -1, D)
    w_e = weights.unsqueeze(-1)
    vacc = torch.zeros(H, n_kept, D, dtype=torch.float32)
    vacc.scatter_add_(1, tgt_e, w_e * ev_)
    wacc = torch.zeros(H, n_kept, dtype=torch.float32)
    wacc.scatter_add_(1, target, weights)

    updated = torch.where(
        (wacc > 0).unsqueeze(-1),
        (kv_ + vacc) / (1.0 + wacc).unsqueeze(-1),
        kv_,
    )
    result = values.clone().float()
    result.scatter_(1, keep_idx_e, updated)
    return result
