"""参考答案：PPO 截断价值损失（clipped value MSE）。"""

import torch


def solve(values_new, values_old, returns, epsilon):
    clipped = torch.clamp(values_new, values_old - epsilon, values_old + epsilon)
    unclipped_loss = (values_new - returns) ** 2
    clipped_loss = (clipped - returns) ** 2
    return torch.maximum(unclipped_loss, clipped_loss).mean()
