import torch


def solve(attn, num_kv_heads):
    """ObservedAttention：对 query 求和 ÷ (L-j) 取平均，再 kv-group 平均。返回 (H, L)。"""
    # TODO
    raise NotImplementedError
