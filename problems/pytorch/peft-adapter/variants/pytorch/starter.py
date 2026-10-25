import torch


def solve(x, W_down, W_up):
    """Adapter 前向：y = relu(x @ W_down.T) @ W_up.T + x。
    x:(...,d) W_down:(r,d) W_up:(d,r)。返回 (...,d)。不加 Dropout；禁止现成 PEFT 库。"""
    # TODO
    raise NotImplementedError
