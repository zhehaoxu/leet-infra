import torch


def solve(pred, target, delta=1.0, reduction="mean"):
    # TODO: Huber 损失。z = |pred-target|；z<=delta 走 0.5*z^2，否则走 delta*(z-0.5*delta)。
    #   用 torch.where 向量化，再按 reduction 归约（mean/sum/none）。未知值抛 ValueError。
    raise NotImplementedError
