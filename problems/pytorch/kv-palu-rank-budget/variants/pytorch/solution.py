"""Reference solution: Palu grouped rank-budget allocation."""

import math

import torch

__all__ = ["solve"]


def solve(fisher, param_ratio_target, head_group_size, lr_group_dims, block_size=32):
    if not torch.is_tensor(fisher) or fisher.dim() != 2:
        raise ValueError("fisher must be a 2-D tensor")
    if not fisher.is_floating_point():
        raise ValueError("fisher must be a floating-point tensor")
    for name, value in (("head_group_size", head_group_size),
                        ("lr_group_dims", lr_group_dims),
                        ("block_size", block_size)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if fisher.shape[0] % head_group_size != 0:
        raise ValueError("fisher rows must be divisible by head_group_size")
    if fisher.shape[1] != lr_group_dims:
        raise ValueError("fisher columns must equal lr_group_dims")
    ratio = float(param_ratio_target)
    if not 0.0 < ratio <= 1.0:
        raise ValueError("param_ratio_target must lie in (0, 1]")

    dtype = fisher.dtype
    w = fisher.to(torch.float64)
    num_groups = w.shape[0] // head_group_size

    # One Fisher scalar per head group: mean over the group's whole block.
    group = w.reshape(num_groups, head_group_size, lr_group_dims).mean(dim=(1, 2))
    fisher_total = group.sum()
    if float(fisher_total) <= 0.0:
        raise ValueError("fisher must have a positive total")

    total_rank = num_groups * lr_group_dims
    target_rank = total_rank * ratio
    rank_float = target_rank * group / fisher_total
    rank_floor = torch.clamp(torch.floor(rank_float), max=float(lr_group_dims))

    # Upstream sorts by ascending remainder (not largest-remainder!) and then
    # walks the list round-robin, skipping groups that already hit the cap.
    order = torch.argsort(rank_float - rank_floor, stable=True)
    need = math.floor(float(target_rank - rank_floor.sum()))
    headroom = int((lr_group_dims - rank_floor).sum())
    if need > headroom:
        raise ValueError("rank budget exceeds available headroom")
    rank_topup = rank_floor.to(torch.int64)
    cursor = 0
    while need > 0:
        g = int(order[cursor % num_groups])
        cursor += 1
        if int(rank_topup[g]) == lr_group_dims:
            continue
        rank_topup[g] += 1
        need -= 1

    # Block alignment uses round-half-to-even and never drops below one block.
    blocks = torch.round(rank_topup.to(torch.float64) / block_size)
    rank_aligned = torch.clamp(blocks, min=1.0).to(torch.int64) * block_size
    rank_sum = rank_aligned.sum()
    stored_ratio = rank_sum.to(torch.float64) / total_rank

    return {
        "rank_float": rank_float.to(dtype),
        "rank_floor": rank_floor.to(torch.int64),
        "rank_topup": rank_topup,
        "rank_aligned": rank_aligned,
        "rank_sum": rank_sum,
        "stored_ratio": stored_ratio.to(dtype),
    }
