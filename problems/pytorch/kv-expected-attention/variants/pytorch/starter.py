import torch


def solve(mu, cov, keys, values, n_sink, epsilon):
    """ExpectedAttention：softmax(mu·k/√d + kᵀΣk/2d) × ||v||，sink 位以 max+1 保底。"""
    # TODO
    raise NotImplementedError
