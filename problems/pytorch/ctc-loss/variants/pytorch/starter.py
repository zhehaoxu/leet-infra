import torch


def solve(log_probs, labels):
    """CTC 负对数似然（前向算法）。log_probs (T,V) 为对数概率，blank=V-1；
    labels (L,) 目标类别（不含 blank）。返回 0 维标量张量 -log p(labels|log_probs)。"""
    # TODO
    raise NotImplementedError
