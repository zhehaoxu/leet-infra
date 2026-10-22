import torch


def solve(x: torch.Tensor, kernel_size, stride=None, padding=0) -> torch.Tensor:
    # TODO: 每个 kernel 窗口取最大值；stride=None 时等于 kernel_size；
    #   padding 填充值必须是 -inf（不是 0）
    raise NotImplementedError
