"""参考答案：DDPM 前向加噪 q(x_t|x_0)。"""

import math

import torch


def solve(x0, eps, alpha_bar):
    return math.sqrt(alpha_bar) * x0 + math.sqrt(1 - alpha_bar) * eps
