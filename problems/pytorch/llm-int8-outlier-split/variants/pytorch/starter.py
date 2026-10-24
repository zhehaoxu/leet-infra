import torch


def solve(W, tau, bits=8):
    """LLM.int8()：列 absmax>tau 作离群列原样保留，其余列 per-tensor 对称 RTN，返回 (recon, mask)。"""
    # TODO
    raise NotImplementedError
