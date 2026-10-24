import torch


def solve(X, W, alpha):
    """SmoothQuant：逐输入通道因子 s_j=max|X_j|^a / max|W_j|^(1-a)，返回 (s, X/s, W*s)。"""
    # TODO
    raise NotImplementedError
