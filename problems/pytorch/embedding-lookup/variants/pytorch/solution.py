"""参考答案：Embedding 查表 —— 按 indices 从 weight 取行，padding_idx 对应位置强制置零。"""

import torch


def solve(weight: torch.Tensor, indices: torch.Tensor, padding_idx: int | None = None) -> torch.Tensor:
    flat = indices.reshape(-1)
    out = weight[flat].reshape(*indices.shape, weight.shape[-1])
    if padding_idx is not None:
        pad_mask = indices == padding_idx
        out = torch.where(pad_mask.unsqueeze(-1), torch.zeros_like(out), out)
    return out
