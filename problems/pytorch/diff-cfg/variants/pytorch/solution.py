"""参考答案：无分类器引导（diffusers 约定 eps_u + w*(eps_c - eps_u)）。"""

import torch


def solve(eps_cond, eps_uncond, w):
    return eps_uncond + w * (eps_cond - eps_uncond)
