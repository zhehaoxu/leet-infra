import torch


def solve(pred, target, reduction="mean"):
    # TODO: 二元交叉熵（输入为概率）。逐元素 -[y*log(p) + (1-y)*log(1-p)]，
    #   先对 p clamp 到 (0,1) 开区间防 log(0)，再按 reduction 归约（mean/sum/none）。
    raise NotImplementedError
