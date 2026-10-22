import torch


def solve(tensors: list[torch.Tensor], dim: int = 0) -> torch.Tensor:
    # TODO: 沿**已存在的维** dim 拼接张量列表：该维长度相加，其余维必须完全相同。
    #   （区别于 stack：stack 会新增一维。）支持负 dim。
    raise NotImplementedError
