"""参考答案：RLOO 留一基线优势。"""

import torch


def _check(rewards):
    if not isinstance(rewards, torch.Tensor):
        raise ValueError("rewards 必须是张量")
    if rewards.dim() != 2:
        raise ValueError("rewards 必须是二维 (B,G)")
    if rewards.dtype not in (torch.float32, torch.float64):
        raise ValueError("rewards 必须是浮点")
    if not bool(torch.isfinite(rewards).all()):
        raise ValueError("rewards 含 NaN/Inf")
    if rewards.shape[0] < 1:
        raise ValueError("组数 B 必须 >= 1")
    if rewards.shape[1] < 2:
        raise ValueError("组内样本数 G 必须 >= 2（留一基线至少需要两个样本）")


def solve(rewards):
    _check(rewards)
    r = rewards.to(torch.float64)
    g = r.shape[1]
    leave_one_out_mean = (r.sum(dim=1, keepdim=True) - r) / (g - 1)
    return r - leave_one_out_mean
