import torch


def solve(logits: torch.Tensor, temperature: float = 1.0, top_p: float | None = None) -> torch.Tensor:
    # TODO: 先 softmax(logits/temperature)（减 max 稳定）；top_p 给定时按降序
    #   累计概率截断 nucleus，剩余项重新归一化（至少保留 1 项）
    raise NotImplementedError
