"""参考答案：构造混合分布 M，返回 beta*KL(P‖M)+(1-beta)*KL(Q‖M) 的逐行均值。"""

import torch


def _kl(A, B):
    mask = A > 0
    term = torch.zeros_like(A)
    term[mask] = A[mask] * (A[mask].log() - B[mask].log())
    return term.sum(dim=-1).mean()


def solve(P, Q, beta=0.5):
    M = beta * P + (1 - beta) * Q
    return beta * _kl(P, M) + (1 - beta) * _kl(Q, M)
