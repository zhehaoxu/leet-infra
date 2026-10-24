import torch


def solve(inputs: list[torch.Tensor], dst: int) -> list[torch.Tensor]:
    # TODO: out[dst] = 各分片按 rank 顺序 cat 的完整结果；out[j!=dst] = inputs[j] 原样保留
    raise NotImplementedError
