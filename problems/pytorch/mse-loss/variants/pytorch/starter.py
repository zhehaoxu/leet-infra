import torch


def solve(pred, target, reduction="mean"):
    # TODO: 逐元素 (pred-target)^2，再按 reduction 归约（mean/sum/none）。
    #   mean=全部元素平均, sum=全部求和, none=返回逐元素张量；未知值抛 ValueError。
    raise NotImplementedError
