"""参考答案：FINCH（kvpress FinchPress）因果归一化窗口注意力打分（向量化版，判题用）。"""

import torch


def solve(attn, num_kv_heads, compression_ratio):
    Hq, W, C = attn.shape
    H = num_kv_heads
    G = Hq // H
    # 行 i（最老的窗口 query 记 i=0）乘以其因果可见 token 数 C + i
    counts = torch.arange(C, C + W, dtype=attn.dtype).view(1, W, 1)
    scaled = attn * counts
    ctx = scaled.mean(dim=1)                  # (Hq, C) 窗口 query 维平均
    ctx = ctx.view(H, G, C).mean(dim=1)       # (H, C)  kv-group 平均
    pad = float(ctx.max()) + 1.0              # 全局 max+1：窗口不可被剪
    scores = torch.cat([ctx, torch.full((H, W), pad, dtype=attn.dtype)], dim=1)
    L = C + W
    n_kept = max(1, int(L * (1 - compression_ratio)))
    order = torch.sort(-scores, dim=-1, stable=True).indices[:, :n_kept]
    kept = torch.sort(order, dim=-1).values
    return scores, kept
