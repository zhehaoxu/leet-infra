"""参考答案：Unfold 膨胀取块 —— unfold 出 kernel 覆盖的整段跨度，再以 dilation 步长抽样。"""

import torch


def solve(x: torch.Tensor, kernel_size: tuple[int, int], dilation: int = 1) -> torch.Tensor:
    kH, kW = kernel_size
    d = dilation
    # 每个输出块在原图上跨越 (k-1)*d+1 个像素：先 unfold 出这段连续跨度
    u = x.unfold(2, (kH - 1) * d + 1, 1).unfold(3, (kW - 1) * d + 1, 1)
    # 再在跨度内每隔 d 取一个，得到 kH×kW 的膨胀采样： (B,C,oh,ow,kH,kW)
    return u[..., ::d, ::d]
