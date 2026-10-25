"""参考答案：DoRA —— 幅值-方向分解的低秩适配前向。"""

import torch


def solve(x, W0, A, B, m, alpha, r):
    s = alpha / r
    V = W0 + s * (B @ A)                              # (d_out, d_in)
    n = V.norm(p=2, dim=1, keepdim=True)             # 每输出行的 L2 范数 (d_out,1)
    Weff = (V / n) * m.unsqueeze(1)                   # 方向 × 幅值，m 沿输入维广播
    return x @ Weff.T                                 # (..., d_out)
