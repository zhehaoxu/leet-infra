"""参考答案：scatter 实现的 one-hot 编码。"""

import torch


def solve(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    if labels.numel() > 0 and not bool(((labels >= 0) & (labels < num_classes)).all()):
        raise ValueError(f"labels 必须落在 [0, {num_classes}) 范围内")
    out = torch.zeros(labels.numel(), num_classes, dtype=torch.float32)
    if labels.numel() > 0:
        out.scatter_(1, labels.unsqueeze(1), 1.0)
    return out
