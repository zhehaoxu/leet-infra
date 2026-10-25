import torch


def solve(P, Q):
    """返回 (forward_kl, reverse_kl)：forward=KL(P‖Q)、reverse=KL(Q‖P)。
    P、Q 为 (N,C) 逐行归一化概率；每行 Σ P log(P/Q)（0log0=0），再对 N 行取均值，
    返回两个 0 维标量张量。"""
    # TODO
    raise NotImplementedError
