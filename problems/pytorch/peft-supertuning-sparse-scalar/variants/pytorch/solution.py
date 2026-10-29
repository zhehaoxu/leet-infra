import torch


def solve(W0, S, k, A, B, alpha, r):
    """SuperTuning / Supra 合并：W' = W0 + topk_mask(|W0|)*S + (r>0)*(alpha/r)*B@A。
    W0,S:(d_out,d_in) A:(r,d_in) B:(d_out,r)。返回 W':(d_out,d_in)。"""
    flat_abs = W0.abs().view(-1)
    _, idx = torch.sort(flat_abs, descending=True, stable=True)
    idx = idx[:k]
    mask = torch.zeros_like(flat_abs)
    mask[idx] = 1.0
    mask = mask.view(W0.shape)
    delta = mask * S
    if r > 0:
        scale = float(alpha) / float(r)
        delta = delta + scale * (B @ A)
    return W0 + delta
