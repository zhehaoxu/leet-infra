"""参考答案：Gather —— 沿指定维用 index 逐元素取行（手写法：meshgrid 高级索引）。"""

import torch


def solve(input: torch.Tensor, index: torch.Tensor, dim: int) -> torch.Tensor:
    if dim < 0:
        dim += input.dim()
    # 对除 dim 外的维度用坐标网格，dim 维用 index 覆盖
    grid = torch.meshgrid(
        *[torch.arange(s) for s in index.shape], indexing="ij"
    )
    gather_index = list(grid)
    gather_index[dim] = index
    return input[tuple(gather_index)]
