import torch


def solve(inputs: list[torch.Tensor]) -> list[torch.Tensor]:
    # TODO: all-to-all —— out[j] = 按源 rank 顺序拼接各 inputs[q] 的第 j 分片（长度 W*k 均分 W 片）
    raise NotImplementedError
