import torch


def solve(x: torch.Tensor, mode: str = "avg") -> torch.Tensor:
    # TODO: 对 dim>=2 的空间维整体求均值/最大值，keepdim=True 保持 (N,C,1,1)
    raise NotImplementedError
