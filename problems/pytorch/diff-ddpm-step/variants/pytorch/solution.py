"""参考答案：DDPM 反向后验一步 p(x_{t-1}|x_t,x_0)。"""

import math

import torch


def solve(x_t, x0_pred, beta_t, alpha_bar_t, alpha_bar_prev, z):
    a_t = 1 - beta_t
    denom = 1 - alpha_bar_t
    c_t = math.sqrt(a_t) * (1 - alpha_bar_prev) / denom
    c_0 = math.sqrt(alpha_bar_prev) * beta_t / denom
    var = beta_t * (1 - alpha_bar_prev) / denom
    return c_t * x_t + c_0 * x0_pred + math.sqrt(var) * z
