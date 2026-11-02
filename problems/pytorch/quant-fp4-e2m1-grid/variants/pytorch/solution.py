import math

import torch

# FP4 E2M1 正半轴网格（含 0）。0.5 是唯一的非零次正规：e=0,m=1 → (1/2)*2^0。
# 下标奇偶恰等于尾数位（0.0=0000 / 0.5=0001 / 1.0=0010 / ... / 6.0=0111），
# 所以 RTNE 的「平局取偶」= 在下标里取偶数那一个。
GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
MAXV = 6.0


def _encode_magnitude(mag):
    """把非负幅值编码成 (下标, 是否平局, 是否饱和)。mag 需已保证非 NaN。"""
    if mag > MAXV:
        return 7, False, True
    k = 0
    while k + 1 < len(GRID) and GRID[k + 1] <= mag:
        k += 1
    if GRID[k] == mag:
        return k, False, False
    lo, hi = k, k + 1
    mid = (GRID[lo] + GRID[hi]) / 2.0
    if mag == mid:
        # 平局：取偶数下标（= 尾数位 0），不是取偶数幅值
        return (lo if lo % 2 == 0 else hi), True, False
    d_lo = mag - GRID[lo]
    d_hi = GRID[hi] - mag
    return (lo if d_lo < d_hi else hi), False, False


def solve(values, block_size, block_scales, global_scale=1.0):
    if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size <= 0:
        raise ValueError("block_size 必须是正整数")
    g = float(global_scale)
    if not (g == g) or g in (float("inf"), float("-inf")) or g <= 0.0:
        raise ValueError("global_scale 必须是有限正数")

    n = int(values.numel())
    num_blocks = (n + block_size - 1) // block_size
    scales = [float(x) for x in block_scales]
    if len(scales) != num_blocks:
        raise ValueError(f"block_scales 长度应为 {num_blocks}，实际 {len(scales)}")
    for s in scales:
        if not (s == s) or s in (float("inf"), float("-inf")) or s <= 0.0:
            raise ValueError("block_scales 每项必须是有限正数")

    flat = values.to(torch.float64).reshape(-1)
    if bool(torch.isnan(flat).any()):
        raise ValueError("values 不得含 NaN（FP4 无 NaN 编码）")

    codes = [0] * n
    dequant = [0.0] * n
    eff_scale = [0.0] * n
    sat_mask = [False] * n
    tie_mask = [False] * n
    block_sat = [0] * num_blocks
    block_ties = [0] * num_blocks

    for i in range(n):
        b = i // block_size
        eff = g * scales[b]
        v = float(flat[i])
        eff_scale[i] = eff
        neg = math.copysign(1.0, v) < 0.0
        scaled = v / eff
        mag = -scaled if scaled < 0.0 else scaled
        idx, tie, sat = _encode_magnitude(mag)
        codes[i] = (8 if neg else 0) | idx
        val = GRID[idx]
        dequant[i] = (-val if neg else val) * eff
        sat_mask[i] = sat
        tie_mask[i] = tie
        if sat:
            block_sat[b] += 1
        if tie:
            block_ties[b] += 1

    return {
        "codes": torch.tensor(codes, dtype=torch.int64),
        "dequant": torch.tensor(dequant, dtype=torch.float64),
        "eff_scale": torch.tensor(eff_scale, dtype=torch.float64),
        "sat_mask": torch.tensor(sat_mask, dtype=torch.bool),
        "tie_mask": torch.tensor(tie_mask, dtype=torch.bool),
        "block_sat": torch.tensor(block_sat, dtype=torch.int64),
        "block_ties": torch.tensor(block_ties, dtype=torch.int64),
        "grid": torch.tensor(GRID, dtype=torch.float64),
    }
