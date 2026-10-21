import torch


class MLPLayer(torch.nn.Module):
    # TODO: 定义 Linear(in -> in*hidden_multiplier) -> ReLU -> Linear(in*hidden_multiplier -> out)
    def __init__(self, in_features: int, hidden_multiplier: int, out_features: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def solve(in_features: int, hidden_multiplier: int, out_features: int) -> torch.nn.Module:
    # TODO: 校验 hidden_multiplier 并返回模块实例
    raise NotImplementedError
