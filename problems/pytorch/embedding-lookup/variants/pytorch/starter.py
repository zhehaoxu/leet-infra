import torch


def solve(weight: torch.Tensor, indices: torch.Tensor, padding_idx: int | None = None) -> torch.Tensor:
    # TODO: 按 indices 从 weight (V, D) 取行，输出形状 (*indices.shape, D)；
    #   若 padding_idx 给定，索引等于 padding_idx 的位置输出整行置零。
    raise NotImplementedError
