import torch


def solve(W, H, bits, damp):
    """GPTQ：逐列量化，量化误差沿 (H+damp·I)^{-1} 的第 i 行摊给未量列补偿，返回重构 W (O, C)。"""
    # TODO
    raise NotImplementedError
