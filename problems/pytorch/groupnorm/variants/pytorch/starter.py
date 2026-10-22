import torch


def solve(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, num_groups: int, eps: float = 1e-5) -> torch.Tensor:
    # TODO: reshape 成 (N, G, C//G, -1)，组内(2,3维)求均值/有偏方差归一化，
    #   还原形状后逐通道乘 weight 加 bias
    raise NotImplementedError
