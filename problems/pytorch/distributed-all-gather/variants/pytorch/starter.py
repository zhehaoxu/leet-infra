import torch


def solve(shards: list[torch.Tensor]) -> list[torch.Tensor]:
    # TODO: 把所有分片按 rank 顺序沿 dim0 拼成 full，返回长度 W 的列表，每项都等于 full
    raise NotImplementedError
