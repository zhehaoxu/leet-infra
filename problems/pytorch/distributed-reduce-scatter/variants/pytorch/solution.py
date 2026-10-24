import torch


def solve(inputs: list[torch.Tensor]) -> list[torch.Tensor]:
    w = len(inputs)
    total = torch.stack(inputs, dim=0).sum(dim=0)   # 逐元素求和，长度 L
    chunk = total.numel() // w
    flat = total.reshape(-1)
    return [flat[i * chunk:(i + 1) * chunk].clone() for i in range(w)]
