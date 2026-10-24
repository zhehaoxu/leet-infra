import torch


def solve(keys):
    """KNorm（kvpress KnormPress）：分数 = -L2 范数。"""
    return -torch.linalg.vector_norm(keys.float(), dim=-1)
