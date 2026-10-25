"""参考答案：LoRA 前向 —— 基座输出 + 低秩旁路两步矩阵乘（不合并权重）。"""

import torch


def solve(x, W0, A, B, alpha, r):
    s = alpha / r
    y0 = x @ W0.T                     # 基座
    down = x @ A.T                    # (..., r) 降维
    y_delta = s * (down @ B.T)        # 升维并缩放
    return y0 + y_delta
