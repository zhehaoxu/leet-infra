"""参考答案：掩膜累加逐行 KL，返回 (前向, 反向) 两个标量。"""

import torch


def _kl(P, Q):
    mask = P > 0
    term = torch.zeros_like(P)
    term[mask] = P[mask] * (P[mask].log() - Q[mask].log())
    return term.sum(dim=-1).mean()


def solve(P, Q):
    return _kl(P, Q), _kl(Q, P)
