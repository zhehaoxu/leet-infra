import torch


def solve(full: torch.Tensor, world: int) -> list[torch.Tensor]:
    flat = full.reshape(-1)
    chunk = flat.numel() // world
    return [flat[i * chunk:(i + 1) * chunk].clone() for i in range(world)]
