"""参考答案：SELU 自归一化激活。

SELU(x) = scale * x                      , x > 0
       = scale * alpha * (exp(x) - 1)     , x <= 0
其中 alpha=1.6732632423543772848170425930144, scale=1.0507009873554804977367086399070。
带固定缩放的小数指数拼接，配合「权重按 SELU 初始化」可让网络自归一化（Klambauer 2017）。
"""

import torch

ALPHA = 1.6732632423543772848170425930144
SCALE = 1.0507009873554804977367086399070


def solve(x):
    return torch.where(x > 0, SCALE * x, SCALE * ALPHA * (torch.exp(x) - 1.0))
