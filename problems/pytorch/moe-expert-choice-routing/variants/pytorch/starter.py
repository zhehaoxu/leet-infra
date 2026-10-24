import torch


def solve(logits, k_c):
    """Expert-Choice：每专家挑 top-k_c 个 token；返回 (mask bool (T,E), renorm_weights (T,E))。"""
    # TODO
    raise NotImplementedError
