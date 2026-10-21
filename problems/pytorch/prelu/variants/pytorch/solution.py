"""参考答案：PReLU —— 负斜率是可广播张量（1D 权重沿通道维 dim=1）。"""

import torch


def solve(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # PReLU 约定：weight 为标量或长度 = 通道数 C 的 1D 张量，作用于 dim=1
    if weight.numel() == 1:
        w = weight.reshape(())
    elif weight.dim() == 1:
        shape = [1] * x.dim()
        shape[1] = weight.shape[0]  # 广播到 (N, C, ...) 的通道维
        w = weight.reshape(shape)
    else:
        w = weight
    return torch.where(x >= 0, x, w * x)
