import torch


def solve(scores: torch.Tensor) -> torch.Tensor:
    # TODO: 对 (..., L, L) 的注意力分数，保留下三角（含对角），
    #   其余位置置 -inf，使 softmax 后未来位置概率为 0
    raise NotImplementedError
