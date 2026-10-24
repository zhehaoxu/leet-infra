import torch


def solve(x: torch.Tensor, Wq: torch.Tensor, w_scale: torch.Tensor) -> torch.Tensor:
    # 先整型权重的矩阵乘，再把 per-channel scale 提到外面按列乘（不物化反量化权重）
    acc = x @ Wq.to(x.dtype).t()          # [M, N]
    return acc * w_scale.to(x.dtype)      # 广播到列（输出通道）
