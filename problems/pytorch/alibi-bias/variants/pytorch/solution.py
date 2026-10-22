"""参考答案：ALiBi —— 用 |i-j| 距离矩阵乘每头斜率，生成注意力位置偏置 (H, L, L)。"""

import torch


def solve(seq_len: int, num_heads: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    # 论文斜率：第 h 头（1-indexed）为 2^(-8h/n)，即从 2^(-8/n) 几何递减到 2^(-8)
    steps = torch.arange(1, num_heads + 1, dtype=dtype)
    slopes = 2.0 ** (-8.0 * steps / num_heads)
    idx = torch.arange(seq_len, dtype=dtype)
    dist = (idx.view(-1, 1) - idx.view(1, -1)).abs()          # (L, L) 距离 |i-j|
    return -slopes.view(-1, 1, 1) * dist                       # (H, L, L)
