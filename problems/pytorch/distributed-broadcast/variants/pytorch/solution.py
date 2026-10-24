import torch


def solve(inputs: list[torch.Tensor], src: int) -> list[torch.Tensor]:
    source = inputs[src]
    return [source.clone() for _ in inputs]
