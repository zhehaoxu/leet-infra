import torch


def solve(indices: torch.Tensor, num_classes: int | None = None) -> torch.Tensor:
    # TODO: 在末尾新增类别维，把 indices 每个位置独热编码；
    #   num_classes=None 时取 max+1，否则用给定值（不足会越界，超出补零）。
    raise NotImplementedError
