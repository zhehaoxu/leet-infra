"""参考答案：One-Hot 编码 —— scatter 到末尾新增的类别维，类别数为 max+1 或显式给定。"""

import torch


def solve(indices: torch.Tensor, num_classes: int | None = None) -> torch.Tensor:
    if num_classes is None:
        num_classes = int(indices.max().item()) + 1 if indices.numel() > 0 else 0
    out = torch.zeros(*indices.shape, num_classes, dtype=torch.long, device=indices.device)
    out.scatter_(-1, indices.unsqueeze(-1).long(), 1)
    return out
