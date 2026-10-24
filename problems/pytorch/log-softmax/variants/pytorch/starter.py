import torch


def solve(x, dim=-1):
    # TODO: 数值稳定 log-softmax。先沿 dim 减最大值 m，再
    # out = (x-m) - log(sum(exp(x-m)))。切勿直接 log(softmax(x))（会溢出）。
    raise NotImplementedError
