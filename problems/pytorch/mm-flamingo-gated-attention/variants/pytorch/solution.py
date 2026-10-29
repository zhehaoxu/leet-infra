"""Immediate-media gated attention only, without the feed-forward sublayer."""

import math

import torch


def solve(x, media, media_locations, wq, wk, wv, wo, gate, n_heads, eps=1e-5):
    b, t, d = x.shape
    if type(n_heads) is not int or n_heads <= 0 or d % n_heads != 0:
        raise ValueError("n_heads must be a positive integer dividing D")
    _, m, k, _ = media.shape
    head_dim = d // n_heads
    text_time = media_locations.to(torch.int64).cumsum(dim=1)
    media_time = torch.arange(1, m + 1, device=x.device)
    visible = (text_time[:, :, None] == media_time[None, None, :])
    visible = visible.unsqueeze(-1).expand(b, t, m, k).reshape(b, t, m * k)

    normalized_x = torch.layer_norm(x, (d,), eps=eps)
    flat_media = media.reshape(b, m * k, d)
    q = (normalized_x @ wq.T).reshape(b, t, n_heads, head_dim).transpose(1, 2)
    keys = (flat_media @ wk.T).reshape(b, m * k, n_heads, head_dim).transpose(1, 2)
    values = (flat_media @ wv.T).reshape(b, m * k, n_heads, head_dim).transpose(1, 2)
    logits = (q @ keys.transpose(-2, -1)) / math.sqrt(head_dim)
    allowed = visible[:, None, :, :]
    has_media = allowed.any(dim=-1, keepdim=True)
    masked_logits = logits.masked_fill(~allowed, -torch.inf)
    safe_logits = torch.where(has_media, masked_logits, torch.zeros_like(masked_logits))
    attention = torch.softmax(safe_logits, dim=-1).masked_fill(~allowed, 0.0)
    merged = (attention @ values).transpose(1, 2).reshape(b, t, d)
    out = x + math.tanh(gate) * (merged @ wo.T)
    return out, attention, visible
