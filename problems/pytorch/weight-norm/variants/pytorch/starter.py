import torch


def solve(weight_g: torch.Tensor, weight_v: torch.Tensor, dim: int = 0) -> torch.Tensor:
    # TODO: 返回 g * v / ||v||_2（沿 dim 求范数，keepdim 广播），
    #   范数需 clamp_min(1e-12) 防止全零切片除零
    raise NotImplementedError
