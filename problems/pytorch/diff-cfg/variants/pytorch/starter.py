import torch


def solve(eps_cond, eps_uncond, w):
    """无分类器引导：eps_guided = eps_uncond + w*(eps_cond - eps_uncond)。
    eps_cond,eps_uncond 同形状，w 为标量 guidance scale。返回同形状 float32。禁止现成扩散库。"""
    # TODO
    raise NotImplementedError
