import torch


def solve(logits, target, alpha):
    """Label smoothing CE：logp=log_softmax(logits)，
    loss=mean((1-alpha)*(-logp[target]) + alpha*(-logp.mean(dim=-1)))。返回 0 维标量张量。"""
    # TODO
    raise NotImplementedError
