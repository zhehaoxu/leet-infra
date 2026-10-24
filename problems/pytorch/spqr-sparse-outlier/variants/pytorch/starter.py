import torch


def solve(W, tau, bits=8):
    """SpQR：逐元素 |W|>tau 作稀疏离群原样保留，其余按逐行 per-row 对称 RTN，返回 (recon, mask)。"""
    # TODO
    raise NotImplementedError
