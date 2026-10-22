import torch


def solve(x: torch.Tensor, output_size) -> torch.Tensor:
    # TODO: 输出第 i 段对应输入的 [floor(i*H/oh), ceil((i+1)*H/oh)) 区间，
    #   每个区间内求均值（宽高各自套同一公式）
    raise NotImplementedError
