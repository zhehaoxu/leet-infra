import torch


def solve(logp_new, logp_old, advantage, epsilon):
    """PPO/GRPO 截断代理目标：ratio=exp(logp_new-logp_old)，
    loss = -mean(min(ratio*A, clip(ratio,1-eps,1+eps)*A))。返回 0 维标量张量。"""
    # TODO
    raise NotImplementedError
