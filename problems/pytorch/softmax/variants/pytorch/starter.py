import torch


def solve(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # TODO: 沿 dim 做数值稳定 softmax：先减 max 再 exp/归一化
    raise NotImplementedError
