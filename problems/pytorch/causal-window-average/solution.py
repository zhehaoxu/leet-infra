"""参考答案：pad + unfold + flip 的因果窗口。"""

import torch
import torch.nn.functional as F


def solve(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    total = kernel.sum()
    if total.item() == 0.0:
        raise ValueError("kernel 的和不能为 0")
    W = kernel.numel()
    padded = F.pad(x, (W - 1, 0))            # 左侧补 W-1 个 0
    windows = padded.unfold(1, W, 1)          # (B, L, W)，窗口内从旧到新
    return (windows * kernel.flip(0)).sum(-1) / total
