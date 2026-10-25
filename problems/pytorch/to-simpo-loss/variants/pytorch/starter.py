import torch


def solve(logp_w, logp_l, len_w, len_l, beta, gamma):
    """SimPO 损失：r_w=logp_w/len_w, r_l=logp_l/len_l;
    loss = -mean(logsigmoid(beta*(r_w - r_l) - gamma))。
    各量形状 (N,)，len 为正。返回 0 维标量。禁止现成对齐库。"""
    # TODO
    raise NotImplementedError
