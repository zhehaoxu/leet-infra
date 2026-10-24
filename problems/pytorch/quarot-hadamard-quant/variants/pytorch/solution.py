import math

import torch


def _hadamard(n):
    """递归构造 n×n 归一化 Hadamard（每元素 ±1/√n），对称且正交。"""
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)


def solve(W, bits):
    """QuaRot：Hadamard 正交旋转摊平离群 → per-tensor 对称 RTN → 反旋转还原。"""
    C = W.shape[1]
    H = _hadamard(C).to(W.dtype)
    Wr = W @ H                                   # 旋转（每行右乘 H）
    qmax = 2 ** (bits - 1) - 1
    scale = float(Wr.abs().max()) or 1.0
    q = torch.clamp(torch.round(Wr / scale * qmax), -qmax - 1, qmax)
    Wrq = q / qmax * scale                        # 旋转域量化→反量化
    return Wrq @ H                                # 反旋转（H 对称正交，H^-1 = H）
