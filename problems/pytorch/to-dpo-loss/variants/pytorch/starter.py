import torch


def solve(chosen_logp, chosen_ref_logp, rejected_logp, rejected_ref_logp, beta):
    """DPO loss：-logsigmoid(beta*((chosen-cchosen_ref)-(rejected-rejected_ref))).mean()。
    log-sigmoid 需数值稳定。返回 0 维标量张量。"""
    # TODO
    raise NotImplementedError
