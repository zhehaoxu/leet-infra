"""参考答案：缩放点积注意力 —— softmax(QK^T/sqrt(d))V，mask 置 -inf，softmax 减 max 稳定。"""

import math

import torch


def solve(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
          mask: torch.Tensor | None = None) -> torch.Tensor:
    d_k = query.shape[-1]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        # torch bool mask 约定：True=允许注意力，False=屏蔽（与 HF 的 attention_mask 相反）
        scores = scores.masked_fill(~mask, float("-inf"))
    shifted = scores - scores.max(dim=-1, keepdim=True).values
    exp = torch.exp(shifted)
    attn = exp / exp.sum(dim=-1, keepdim=True)
    # 整行被屏蔽时上面会产生 NaN，按 torch 约定输出 0
    attn = torch.nan_to_num(attn, nan=0.0)
    return torch.matmul(attn, value)
