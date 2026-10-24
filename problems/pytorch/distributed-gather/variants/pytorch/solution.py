import torch


def solve(inputs: list[torch.Tensor], dst: int) -> list[torch.Tensor]:
    full = torch.cat(inputs, dim=0)
    out = []
    for j in range(len(inputs)):
        if j == dst:
            out.append(full.clone())
        else:
            out.append(inputs[j].clone())   # 非目标 rank 原样保留
    return out
