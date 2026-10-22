"""参考答案：因果掩码 —— 下三角保留，其上位置置 -inf（自回归模型「不看未来」）。"""

import torch


def solve(scores: torch.Tensor) -> torch.Tensor:
    seq_len = scores.shape[-1]
    keep = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=scores.device))
    return scores.masked_fill(~keep, float("-inf"))
