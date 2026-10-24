import torch


def solve(logits, target, reduction="mean"):
    # TODO: 交叉熵 = -log softmax(logits)[target]。
    #   先做数值稳定的 log-softmax（减每行最大值），再按 target gather 取负，
    #   最后按 reduction 归约（mean 对 N 个样本平均 / sum / none）。未知值抛 ValueError。
    raise NotImplementedError
