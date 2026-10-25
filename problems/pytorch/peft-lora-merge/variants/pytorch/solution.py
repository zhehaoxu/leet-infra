"""参考答案：LoRA 权重合并 —— 显式 W' = W0 + (alpha/r) (B @ A)。"""

import torch


def solve(W0, A, B, alpha, r):
    s = alpha / r
    return W0 + s * (B @ A)
