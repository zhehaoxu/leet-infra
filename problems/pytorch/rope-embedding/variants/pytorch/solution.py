"""参考答案：RoPE —— 相邻两维看作复数，按 pos * theta_i 旋转；等价于成对的 2D 旋转矩阵。"""

import torch


def solve(x: torch.Tensor, positions: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    d = x.shape[-1]
    half = d // 2
    # theta_i = base^(-2i/d)，i = 0..d/2-1
    idx = torch.arange(half, dtype=x.dtype, device=x.device) * 2.0
    theta = base ** (-idx / d)
    ang = positions.to(x.dtype).view(-1, 1) * theta.view(1, -1)   # (L, half)
    cos, sin = ang.cos(), ang.sin()
    x1 = x[..., 0::2]   # 偶数维
    x2 = x[..., 1::2]   # 奇数维
    rotated = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return rotated.flatten(-2, -1)
