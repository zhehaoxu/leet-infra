import torch


def solve(inputs: list[torch.Tensor], root: int) -> list[torch.Tensor]:
    total = torch.stack(inputs, dim=0).sum(dim=0)
    out = [t.clone() for t in inputs]
    out[root] = total
    return out
