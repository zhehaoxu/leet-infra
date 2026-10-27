import torch


def solve(W0, w1, w2, alpha, r):
    """LoKr 合并：W' = W0 + (alpha/r) * kron(w1, w2)。
    W0:(a*c, b*d) w1:(a,b) w2:(c,d)。返回 W':(a*c, b*d)。禁止现成 PEFT 库。"""
    s = float(alpha) / float(r)
    return W0 + torch.kron(w1, w2) * s
