import torch


def solve(P, Q, beta=0.5):
    """Jensen-Shannon 散度：M=beta*P+(1-beta)*Q，
    逐行 JSD=beta*KL(P‖M)+(1-beta)*KL(Q‖M)（0log0=0），再对 N 行取均值，返回 0 维标量。"""
    # TODO
    raise NotImplementedError
