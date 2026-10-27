import torch


def solve(W, bits, iters):
    """HQQ 半二次闭式 scale（float64 内部、float32 输出）。"""
    if not (2 <= int(bits) <= 8):
        raise ValueError("bits 必须在 [2, 8] 内")
    if int(iters) < 0:
        raise ValueError("iters 必须非负")
    qmax = 2 ** int(bits) - 1
    scales, zeros, outs = [], [], []
    for g in range(W.shape[0]):
        w = W[g].to(torch.float64)
        lo, hi = float(w.min()), float(w.max())
        if hi <= lo:
            scales.append(1.0)
            zeros.append(0)
            outs.append(torch.full_like(w, lo))
            continue
        sc = (hi - lo) / qmax
        zp = int(min(max(round(-lo / sc), 0), qmax))
        q = ((w / sc).round() + zp).clamp(0, qmax) - zp
        for _ in range(int(iters)):
            den = float((q * q).sum())
            if den > 1e-12:
                sc = float((w * q).sum()) / den
            q = ((w / sc).round() + zp).clamp(0, qmax) - zp
        # 收尾：让返回的 scale 与返回的 q 严格满足闭式关系
        den = float((q * q).sum())
        if den > 1e-12:
            sc = float((w * q).sum()) / den
        scales.append(sc)
        zeros.append(zp)
        outs.append(q * sc)
    return (torch.tensor(scales, dtype=torch.float32),
            torch.tensor(zeros, dtype=torch.int64),
            torch.stack(outs).to(torch.float32))
