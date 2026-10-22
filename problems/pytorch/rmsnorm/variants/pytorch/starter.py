import torch


def solve(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # TODO: y = x / sqrt(mean(x^2, dim=-1, keepdim) + eps) * weight（不减均值、不加 bias）
    raise NotImplementedError
