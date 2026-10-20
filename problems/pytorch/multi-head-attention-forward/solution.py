"""参考答案：reshape/transpose 拆分多头的 attention 前向。"""

import math

import torch
import torch.nn.functional as F


def solve(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
          num_heads: int, causal: bool = False) -> torch.Tensor:
    B, L, D = q.shape
    if D % num_heads != 0:
        raise ValueError(f"D={D} 不能被 num_heads={num_heads} 整除")
    d = D // num_heads

    def split(t: torch.Tensor) -> torch.Tensor:
        return t.view(B, L, num_heads, d).transpose(1, 2)  # (B, H, L, d)

    qh, kh, vh = split(q), split(k), split(v)
    scores = qh @ kh.transpose(-1, -2) / math.sqrt(d)
    if causal:
        mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=q.device))
        scores = scores.masked_fill(~mask, float("-inf"))
    att = F.softmax(scores, dim=-1)
    out = att @ vh
    return out.transpose(1, 2).reshape(B, L, D)
