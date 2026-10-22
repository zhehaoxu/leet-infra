import torch


def solve(x: torch.Tensor, kernel_size: tuple[int, int], dilation: int = 1) -> torch.Tensor:
    # TODO: 输入 (B,C,H,W)，抽出每个 kernel 位置的感受野块，输出 (B,C,oh,ow,kH,kW)。
    #   块 [r,w] 采样 x[..., r + i*dilation, w + j*dilation]（i<kH, j<kW）。
    #   oh = H-(kH-1)*dilation, ow = W-(kW-1)*dilation。可用 Tensor.unfold 实现。
    raise NotImplementedError
