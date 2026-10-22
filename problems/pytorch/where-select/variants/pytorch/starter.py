import torch


def solve(condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # TODO: elementwise 三目：condition 为真处取 x，否则取 y；
    #   condition/x/y 需按广播规则对齐（支持标量与不同形状广播）。
    raise NotImplementedError
