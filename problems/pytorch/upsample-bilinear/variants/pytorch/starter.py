import torch


def solve(x: torch.Tensor, size, align_corners: bool = False) -> torch.Tensor:
    # TODO: 目标像素映射回源坐标（align_corners=False: src=(dst+0.5)*scale-0.5;
    #   True: src=dst*(in-1)/(out-1)），clamp 后取四邻域双线性混合
    raise NotImplementedError
