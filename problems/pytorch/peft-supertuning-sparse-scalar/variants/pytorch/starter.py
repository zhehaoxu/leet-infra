import torch


def solve(W0, S, k, A, B, alpha, r):
    """SuperTuning / Supra 合并：W' = W0 + topk_mask(|W0|)*S + (r>0)*(alpha/r)*B@A。
    W0,S:(d_out,d_in) A:(r,d_in) B:(d_out,r)。返回 W':(d_out,d_in)。"""
    # TODO
    raise NotImplementedError
