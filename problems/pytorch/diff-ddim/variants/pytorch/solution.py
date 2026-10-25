"""参考答案：DDIM 确定性去噪一步（sigma_t=0）。"""

import math

import torch


def solve(x_t, eps_pred, alpha_bar_t, alpha_bar_prev):
    x0_hat = (x_t - math.sqrt(1 - alpha_bar_t) * eps_pred) / math.sqrt(alpha_bar_t)
    return math.sqrt(alpha_bar_prev) * x0_hat + math.sqrt(1 - alpha_bar_prev) * eps_pred
