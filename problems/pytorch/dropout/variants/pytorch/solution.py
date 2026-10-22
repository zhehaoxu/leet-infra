"""参考答案：Inverted Dropout —— 独立 Generator 锁 seed 采样掩码，train 时除以 1-p。"""

import torch


def solve(x: torch.Tensor, p: float = 0.5, training: bool = True, seed: int = 1234) -> torch.Tensor:
    if not training or p <= 0.0:
        return x
    g = torch.Generator(device=x.device).manual_seed(seed)
    mask = (torch.rand(x.shape, generator=g, device=x.device, dtype=x.dtype) >= p).to(x.dtype)
    return x * mask / (1.0 - p)
