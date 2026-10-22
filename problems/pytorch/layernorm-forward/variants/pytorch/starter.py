import torch


def solve(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    # TODO: 沿 x 的尾部 weight.dim() 个维求均值/有偏方差，
    #   (x-mu)/sqrt(var+eps)*weight+bias
    raise NotImplementedError
