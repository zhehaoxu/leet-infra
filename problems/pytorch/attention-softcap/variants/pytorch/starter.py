import torch


def solve(scores, cap):
    # TODO: 先做 soft-capping 再 softmax：
    # capped = cap * tanh(scores / cap)；返回对最后一维的 softmax(capped)。
    raise NotImplementedError
