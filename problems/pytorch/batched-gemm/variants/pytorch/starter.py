import torch


def solve(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # TODO: out[...,i,j] = Σ_k a[...,i,k]*b[...,k,j]；批维按 torch.matmul
    #   语义广播（允许一侧为 2D 或 batch 维为 1），不得调用 matmul/bmm/@/dot
    raise NotImplementedError
