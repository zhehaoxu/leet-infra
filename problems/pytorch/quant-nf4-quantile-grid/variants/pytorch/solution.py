import torch


def solve(w, offset):
    """NF4 分位码本构造 + 64 宽块编码（float64 内部、float32 输出）。"""
    if not (0.5 < float(offset) < 1.0):
        raise ValueError("offset 必须在 (0.5, 1) 内")
    if w.shape[1] % 64 != 0:
        raise ValueError("最后一维必须是 64 的倍数")

    off = float(offset)
    pos = [off + (0.5 - off) * i / 8.0 for i in range(8)]
    neg = [off + (0.5 - off) * i / 7.0 for i in range(7)]
    vals = [float(torch.erfinv(torch.tensor(2.0 * p - 1.0, dtype=torch.float64)) * 2 ** 0.5)
            for p in pos]
    vals += [-float(torch.erfinv(torch.tensor(2.0 * p - 1.0, dtype=torch.float64)) * 2 ** 0.5)
             for p in neg]
    vals.append(0.0)
    table = torch.tensor(sorted(vals), dtype=torch.float64)
    table = table / table.abs().max()

    w64 = w.to(torch.float64)
    m, n = w64.shape
    blocks = w64.view(m, n // 64, 64)
    scale = blocks.abs().amax(dim=2, keepdim=True)            # (m, n//64, 1)
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    codes = (blocks / safe).unsqueeze(-1).sub(table).abs().argmin(dim=-1)
    codes = torch.where(scale > 0, codes, torch.full_like(codes, 7))
    deq = table[codes] * scale
    return (table.to(torch.float32), codes.reshape(m, n), deq.reshape(m, n).to(torch.float32))
