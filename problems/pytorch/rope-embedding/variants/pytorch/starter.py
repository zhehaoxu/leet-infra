import torch


def solve(x: torch.Tensor, positions: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    # TODO: 把 (x[2i], x[2i+1]) 按角度 positions*base^(-2i/d) 做 2D 旋转；
    #   x 形状 (..., L, d)，positions 形状 (L,)
    raise NotImplementedError
