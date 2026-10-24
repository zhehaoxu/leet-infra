import torch
import torch.nn.functional as F


def solve(attn, num_kv_heads):
    """对齐 kvpress TOVAPress.score（无 batch）。"""
    # attn shape (Hq, L, L)
    last = attn[:, -1:, :-1]                # (Hq, 1, L-1)
    shared = last.mean(dim=0)                # (1, L-1) — 跨 Hq 平均
    out = shared.repeat(num_kv_heads, 1)     # (H, L-1)
    m = out.max().item() + 1.0
    return F.pad(out, (0, 1), value=m)       # (H, L)
