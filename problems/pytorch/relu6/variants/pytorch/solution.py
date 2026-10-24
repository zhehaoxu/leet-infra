"""参考答案：ReLU6 = min(max(x, 0), 6)。

在 ReLU 基础上再加一个上界 6，把激活压到 [0, 6]。有界性让量化更友好（动态范围固定），
是 MobileNet 等移动端网络的原用激活（论文里为 int8 推理特意选它）。
"""

import torch


def solve(x):
    return torch.clamp(x, min=0.0, max=6.0)
