import torch


def solve(prob, target, smoothing=1.0, reduction="mean"):
    """Soft Dice 损失：逐样本沿最后一维 loss=1-(2*Σpy+s)/(Σp+Σy+s)，
    reduction mean/sum/none。非法 reduction 抛 ValueError。"""
    # TODO
    raise NotImplementedError
