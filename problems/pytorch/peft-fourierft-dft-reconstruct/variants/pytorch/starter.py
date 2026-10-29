import torch


def solve(W0, freqs, coeffs, alpha):
    """FourierFT 合并：W' = W0 + alpha * sum_{(u,v)} c * cos(2*pi*(u*i/M+v*j/N))。
    W0:(d_out,d_in) freqs:(m,2) coeffs:(m,)。返回 W':(d_out,d_in)。"""
    # TODO
    raise NotImplementedError
