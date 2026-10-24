import torch


def solve(shards: list[torch.Tensor], max_norm: float, norm_type: float = 2.0):
    p = float(norm_type)
    # 跨所有 rank 汇总 |g|^p —— 这一步就是「全局 reduce」，不能各 rank 自算
    total_pow = sum((g.abs() ** p).sum().item() for g in shards)
    total_norm = total_pow ** (1.0 / p)
    if total_norm == 0.0 or total_norm <= float(max_norm):
        scale = 1.0
    else:
        scale = float(max_norm) / total_norm
    clipped = [g * scale for g in shards]      # 新张量，不原地改输入
    return clipped, total_norm
