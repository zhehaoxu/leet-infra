import torch


def solve(logits, target, gamma=2.0, alpha=None, reduction="mean"):
    """Focal Loss: -(1-p_t)^gamma * log p_t（p_t 为 softmax 真类概率），
    alpha 非 None 时作标量权重，reduction mean/sum/none。非法 reduction 抛 ValueError。"""
    # TODO
    raise NotImplementedError
