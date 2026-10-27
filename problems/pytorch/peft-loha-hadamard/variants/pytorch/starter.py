import torch


def solve(W0, w1_a, w1_b, w2_a, w2_b, alpha, r):
    """LoHa 合并：W' = W0 + (alpha/r) * (w1_a @ w1_b ⊙ w2_a @ w2_b)。
    W0:(d_out,d_in) w1_a,w2_a:(d_out,r) w1_b,w2_b:(r,d_in)。返回 W':(d_out,d_in)。禁止现成 PEFT 库。"""
    # TODO
    raise NotImplementedError
