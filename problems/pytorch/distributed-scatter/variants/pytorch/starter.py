import torch


def solve(full: torch.Tensor, world: int) -> list[torch.Tensor]:
    # TODO: 把 full 均分 world 段（chunk=len/world），rank i 取第 i 段，返回长度 world 的列表
    raise NotImplementedError
