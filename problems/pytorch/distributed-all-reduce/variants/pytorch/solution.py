import torch


def solve(inputs: list[torch.Tensor]) -> list[torch.Tensor]:
    total = torch.stack(inputs, dim=0).sum(dim=0)
    return [total.clone() for _ in inputs]
