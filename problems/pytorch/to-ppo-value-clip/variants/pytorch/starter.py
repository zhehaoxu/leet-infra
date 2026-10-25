import torch


def solve(values_new, values_old, returns, epsilon):
    """PPO 截断价值损失：c=clip(values_new, values_old±epsilon);
    loss=mean(max((values_new-returns)^2,(c-returns)^2))。返回 0 维标量张量。禁止现成 RL 库。"""
    # TODO
    raise NotImplementedError
