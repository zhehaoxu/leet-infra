import torch


def solve(input: torch.Tensor, index: torch.Tensor, dim: int) -> torch.Tensor:
    # TODO: 实现 torch.gather 语义，输出形状与 index 相同：
    #   out[i][j][k] = input[ index[i][j][k] ][j][k]（当 dim=0）；
    #   非 dim 维坐标与输出一致，index 与 input 维数相同。
    raise NotImplementedError
