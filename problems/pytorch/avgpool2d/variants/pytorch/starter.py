import torch


def solve(x: torch.Tensor, kernel_size, stride=None, padding=0, count_include_pad: bool = True) -> torch.Tensor:
    # TODO: 每个窗口求均值；count_include_pad=False 时分母只算
    #   非 padding 的有效像素数（边界窗口的分母会变小）
    raise NotImplementedError
