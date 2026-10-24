import torch


def solve(logits):
    """PaLM/ST-MoE router z-loss。"""
    ls = torch.logsumexp(logits.float(), dim=-1)
    return (ls ** 2).mean()
