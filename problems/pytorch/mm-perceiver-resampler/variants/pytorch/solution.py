"""One masked Perceiver latent-attention residual step, not a full resampler."""

import math

import torch


def solve(visual, latents, visual_mask, wq, wk, wv, wo, n_heads, eps=1e-5):
    b, k, d = latents.shape
    if type(n_heads) is not int or n_heads <= 0 or d % n_heads != 0:
        raise ValueError("n_heads must be a positive integer dividing D")
    head_dim = d // n_heads
    normalized_visual = torch.layer_norm(visual, (d,), eps=eps)
    normalized_latents = torch.layer_norm(latents, (d,), eps=eps)
    source = torch.cat((normalized_visual, normalized_latents), dim=1)
    length = source.shape[1]

    q = (normalized_latents @ wq.T).reshape(b, k, n_heads, head_dim).transpose(1, 2)
    keys = (source @ wk.T).reshape(b, length, n_heads, head_dim).transpose(1, 2)
    values = (source @ wv.T).reshape(b, length, n_heads, head_dim).transpose(1, 2)
    visible = torch.cat(
        (visual_mask, torch.ones((b, k), dtype=torch.bool, device=visual.device)), dim=1
    )
    logits = (q @ keys.transpose(-2, -1)) / math.sqrt(head_dim)
    attention = torch.softmax(logits.masked_fill(~visible[:, None, None, :], -torch.inf), dim=-1)
    merged = (attention @ values).transpose(1, 2).reshape(b, k, d)
    output = latents + merged @ wo.T
    return output, attention
