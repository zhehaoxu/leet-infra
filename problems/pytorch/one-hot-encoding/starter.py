import torch


def solve(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """把 (N,) 的类别下标转成 (N, num_classes) 的 float32 one-hot 矩阵。

    - label 越界（<0 或 >=num_classes）抛 ValueError
    - 空批次返回 (0, num_classes) 空张量
    - 禁止使用 F.one_hot
    """
    # TODO: 越界校验 -> zeros -> scatter/高级索引置 1
    raise NotImplementedError("请实现 solve()")
