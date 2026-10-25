"""参考答案：GRPO 组内标准化优势。"""

import torch


def solve(rewards, group_size):
    x = rewards.view(-1, group_size)
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True, unbiased=False)
    return ((x - mean) / std).reshape(-1)
