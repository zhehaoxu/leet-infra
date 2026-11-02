import math

import torch


def _as_bins(bins, dim):
    if isinstance(bins, int) and not isinstance(bins, bool):
        if bins < 1:
            raise ValueError("bins 必须是 >=1 的整数")
        return [bins] * dim
    if isinstance(bins, (list, tuple)):
        if len(bins) != dim:
            raise ValueError("bins 长度必须等于描述子维度")
        out = []
        for b in bins:
            if not isinstance(b, int) or isinstance(b, bool) or b < 1:
                raise ValueError("bins 每项必须是 >=1 的整数")
            out.append(b)
        return out
    raise ValueError("bins 必须是整数或整数列表")


def solve(descriptors, scores, bins, ranges):
    if descriptors.dim() != 2:
        raise ValueError("descriptors 形状必须是 (N, D)")
    n, dim = descriptors.shape
    if dim < 1:
        raise ValueError("描述子维度必须 >=1")
    if scores.dim() != 1 or scores.numel() != n:
        raise ValueError("scores 长度必须等于 N")
    b = _as_bins(bins, dim)
    if len(ranges) != dim:
        raise ValueError("ranges 长度必须等于描述子维度")
    lo, hi = [], []
    for r in ranges:
        if len(r) != 2:
            raise ValueError("ranges 每项必须是 (lo, hi)")
        a, c = float(r[0]), float(r[1])
        if not (a < c):
            raise ValueError("ranges 每项必须满足 lo < hi")
        lo.append(a)
        hi.append(c)
    if bool(torch.isnan(scores).any()):
        raise ValueError("scores 不得含 NaN")

    total = 1
    for x in b:
        total *= x
    elite_score = [float("-inf")] * total
    elite_index = [-1] * total
    cells, accepted = [], []
    desc = descriptors.to(torch.float64).tolist()
    sc = scores.to(torch.float64).tolist()

    for i in range(n):
        d = desc[i]
        idx, outside = [], False
        for k in range(dim):
            v = d[k]
            if not (lo[k] <= v <= hi[k]):
                outside = True
                break
            step = (v - lo[k]) / (hi[k] - lo[k])
            slot = int(math.floor(step * b[k]))
            if slot >= b[k]:
                slot = b[k] - 1
            idx.append(slot)
        if outside:
            cells.append(-1)
            accepted.append(False)
            continue
        flat = 0
        for k in range(dim):
            flat = flat * b[k] + idx[k]
        cells.append(flat)
        if sc[i] > elite_score[flat]:
            elite_score[flat] = sc[i]
            elite_index[flat] = i
            accepted.append(True)
        else:
            accepted.append(False)

    filled = [i for i in range(total) if elite_index[i] >= 0]
    best_score, best_index = float("-inf"), -1
    for i in filled:
        if elite_score[i] > best_score:
            best_score, best_index = elite_score[i], elite_index[i]

    return {
        "cells": torch.tensor(cells, dtype=torch.int64),
        "accepted": torch.tensor(accepted, dtype=torch.bool),
        "elite_score": torch.tensor(elite_score, dtype=torch.float64),
        "elite_index": torch.tensor(elite_index, dtype=torch.int64),
        "occupancy": len(filled),
        "best_score": best_score,
        "best_index": best_index,
    }
