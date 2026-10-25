import torch


def solve(x0, eps, alpha_bar):
    """DDPM 前向加噪：x_t = sqrt(alpha_bar)*x0 + sqrt(1-alpha_bar)*eps。
    x0,eps 同形状，alpha_bar 为标量 (0,1]。返回与 x0 同形状的 float32 张量。禁止现成扩散库。"""
    # TODO
    raise NotImplementedError
