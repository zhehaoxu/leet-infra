"""参考答案：结构化丢弃（DropPath / Dropout2d / Dropout3d）—— 显式 mask + inverted scaling。"""

import math

import torch


def solve(x, p, mask, mode="droppath"):
    pf = float(p)
    if not math.isfinite(pf) or pf < 0.0 or pf >= 1.0:
        raise ValueError(f"p 必须满足 0 <= p < 1，got {p!r}")
    if not torch.is_tensor(mask):
        raise ValueError(f"mask 必须是 torch 张量，got {type(mask)!r}")

    if mode == "droppath":
        if x.dim() < 1:
            raise ValueError(f"droppath 要求 x 秩 >= 1，got {tuple(x.shape)}")
        b = x.shape[0]
        # (B,) 或 (B, 1, ..., 1)（1 的个数 = x 维数 - 1），广播后恰为 x 形状
        allowed = {(b,), (b,) + (1,) * (x.dim() - 1)}
        m_shape = (b,) + (1,) * (x.dim() - 1)
    elif mode == "dropout2d":
        if x.dim() != 4:
            raise ValueError(f"dropout2d 要求 x 恰 4 维 (N,C,H,W)，got {tuple(x.shape)}")
        allowed = {(x.shape[0], x.shape[1]), (x.shape[0], x.shape[1], 1, 1)}
        m_shape = (x.shape[0], x.shape[1], 1, 1)
    elif mode == "dropout3d":
        if x.dim() != 5:
            raise ValueError(f"dropout3d 要求 x 恰 5 维 (N,C,D,H,W)，got {tuple(x.shape)}")
        allowed = {(x.shape[0], x.shape[1]), (x.shape[0], x.shape[1], 1, 1, 1)}
        m_shape = (x.shape[0], x.shape[1], 1, 1, 1)
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    if tuple(mask.shape) not in allowed:
        raise ValueError(f"mode={mode} 下 mask 形状 {tuple(mask.shape)} 不合法，"
                         f"允许：{sorted(allowed)}（广播后必须与 x 形状完全一致）")
    m = mask.reshape(m_shape).to(dtype=x.dtype)
    scale = 1.0 / (1.0 - pf)
    return x * m * scale
