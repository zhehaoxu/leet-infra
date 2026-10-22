"""参考答案：多头注意力 —— (B,L,D) 拆成 (B,H,L,dh)，逐头做缩放点积注意力，再合回 D。"""

import math

import torch


def solve(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int,
          mask: torch.Tensor | None = None) -> torch.Tensor:
    b, lq, d = q.shape
    lk = k.shape[1]
    assert d % num_heads == 0, "特征维必须被头数整除"
    dh = d // num_heads
    # 关键两步：view 切出头维 -> transpose 把 head 挪到 batch 后
    qh = q.view(b, lq, num_heads, dh).transpose(1, 2)      # (B,H,Lq,dh)
    kh = k.view(b, lk, num_heads, dh).transpose(1, 2)      # (B,H,Lk,dh)
    vh = v.view(b, lk, num_heads, dh).transpose(1, 2)
    scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(dh)
    if mask is not None:
        # torch bool mask 约定：True=允许注意力，False=屏蔽
        scores = scores.masked_fill(~mask, float("-inf"))
    shifted = scores - scores.max(dim=-1, keepdim=True).values
    exp = torch.exp(shifted)
    attn = exp / exp.sum(dim=-1, keepdim=True)
    attn = torch.nan_to_num(attn, nan=0.0)                # 整行屏蔽时输出 0
    ctx = torch.matmul(attn, vh)                            # (B,H,Lq,dh)
    return ctx.transpose(1, 2).reshape(b, lq, d)           # 合并头
