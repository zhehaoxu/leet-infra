import torch


def solve(x, W, lam):
    """IA3 前向：y = lam * (x @ W.T)。x:(...,d_in) W:(d_out,d_in) lam:(d_out,) 广播到最后维。
    返回 (...,d_out)。禁止现成 PEFT 库。"""
    # TODO
    raise NotImplementedError
