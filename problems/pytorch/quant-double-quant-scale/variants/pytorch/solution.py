import torch


def solve(scales, bits, block_size):
    """double quantization 的 scale 二级量化（float64 内部、float32 输出）。"""
    if not (2 <= int(bits) <= 8):
        raise ValueError("bits 必须在 [2, 8] 内")
    if int(block_size) <= 0:
        raise ValueError("block_size 必须为正")
    qmax = 2 ** (int(bits) - 1) - 1
    s64 = scales.to(torch.float64).reshape(-1)
    b = int(s64.numel())
    bs = int(block_size)
    n_grp = (b + bs - 1) // bs
    codes = torch.zeros(b, dtype=torch.int64)
    gs = []
    deq = torch.zeros(b, dtype=torch.float64)
    for g in range(n_grp):
        lo, hi = g * bs, min((g + 1) * bs, b)
        blk = s64[lo:hi]
        mx = float(blk.abs().max())
        s2 = mx / qmax if mx > 0 else 1.0
        c = (blk / s2).round().clamp(-qmax, qmax)
        codes[lo:hi] = c.to(torch.int64)
        deq[lo:hi] = c * s2
        gs.append(s2)
    return (codes, torch.tensor(gs, dtype=torch.float32), deq.to(torch.float32))
