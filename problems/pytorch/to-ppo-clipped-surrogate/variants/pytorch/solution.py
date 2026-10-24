import torch


def solve(logp_new, logp_old, advantage, epsilon):
    """PPO/GRPO 截断代理目标。"""
    ratio = torch.exp(logp_new - logp_old)
    clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    obj = torch.min(ratio * advantage, clipped * advantage)
    return -obj.mean()
