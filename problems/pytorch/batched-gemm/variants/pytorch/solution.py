"""参考答案：批量矩阵乘 —— 广播批维后用逐元素外积求和实现 matmul 语义。"""

import torch


def solve(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.dim() < 2 or b.dim() < 2:
        raise ValueError("仅支持 2D 及以上输入")
    if a.shape[-1] != b.shape[-2]:
        raise ValueError(f"内维不匹配: {tuple(a.shape)} @ {tuple(b.shape)}")
    batch = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    a_e = a.expand(*batch, a.shape[-2], a.shape[-1])
    b_e = b.expand(*batch, b.shape[-2], b.shape[-1])
    # (...,n,m,1) * (...,1,m,p) 沿 m 求和 = 逐元素定义的矩阵乘
    return (a_e.unsqueeze(-1) * b_e.unsqueeze(-3)).sum(dim=-2)
