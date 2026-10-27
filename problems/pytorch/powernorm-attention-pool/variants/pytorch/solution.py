import math

import torch


def solve(x, mode="powernorm", seed=None, eps=1e-6):
    """PowerNorm 与单种子注意力池化，契约见 starter.py docstring。"""
    if mode not in ("powernorm", "attn-pool"):
        raise ValueError(f"mode 仅支持 'powernorm' / 'attn-pool'，got {mode!r}")
    if not isinstance(x, torch.Tensor) or x.dtype not in (torch.float32, torch.float64):
        raise ValueError("x 必须是 float32 / float64 张量")

    if mode == "powernorm":
        if x.dim() not in (3, 4):
            raise ValueError(f"powernorm 需要 (N,C,L) 3-D 或 (N,C,H,W) 4-D 输入，got {x.dim()}-D")
        if not isinstance(eps, (int, float)) or isinstance(eps, bool) or eps <= 0:
            raise ValueError(f"eps 必须为正数，got {eps!r}")
        flat = x.reshape(x.shape[0], x.shape[1], -1)
        # 通道均值平方（不做减均值/中心化，纯二阶矩）
        ms = flat.pow(2).mean(dim=1, keepdim=True)
        y = flat / torch.sqrt(ms + eps)
        return y.reshape(x.shape)

    # attn-pool
    if x.dim() != 3:
        raise ValueError(f"attn-pool 需要 (B,L,D) 3-D 输入，got {x.dim()}-D")
    if seed is None:
        raise ValueError("attn-pool 必须提供 seed（(D,) 种子/查询向量），got None")
    if not isinstance(seed, torch.Tensor) or seed.dim() != 1:
        raise ValueError("seed 必须是一维 (D,) 张量")
    d = x.shape[2]
    if seed.shape[0] != d:
        raise ValueError(f"seed 长度应等于 D={d}，got {seed.shape[0]}")
    q = seed.to(x.dtype)
    logits = x @ q / math.sqrt(d)          # (B,L)
    attn = torch.softmax(logits, dim=-1)   # 沿 L 维，数值稳定
    return (attn.unsqueeze(1) @ x).squeeze(1)  # (B,1,L)@(B,L,D) -> (B,D)
