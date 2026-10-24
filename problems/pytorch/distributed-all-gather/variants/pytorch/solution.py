import torch


def solve(shards: list[torch.Tensor]) -> list[torch.Tensor]:
    full = torch.cat(shards, dim=0)
    return [full.clone() for _ in shards]
