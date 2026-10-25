"""参考答案：Adapter —— 瓶颈降维→ReLU→升维→残差（行向量批量惯例）。"""

import torch


def solve(x, W_down, W_up):
    down = torch.relu(x @ W_down.T)     # (..., r)
    up = down @ W_up.T                  # (..., d)
    return up + x                       # 残差
