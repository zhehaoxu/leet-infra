import torch


def solve(logits, k_c):
    """Shazeer 2022 Expert-Choice Routing。"""
    probs = torch.softmax(logits.float(), dim=-1)
    T, E = probs.shape
    mask = torch.zeros(T, E, dtype=torch.bool)
    for e in range(E):
        _, top = torch.topk(probs[:, e], k_c, dim=0)
        mask[top, e] = True
    weights = probs * mask.float()
    s = weights.sum(dim=-1, keepdim=True)
    out = torch.where(s > 0, weights / s.clamp_min(1e-30), torch.zeros_like(weights))
    return mask, out
