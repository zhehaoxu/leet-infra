import torch


def solve(W):
    """BlockFP：1×128 tile + UE8M0 共享指数编码。"""
    M, N = W.shape
    if N % 128 != 0:
        raise ValueError("N 必须是 128 的倍数")
    tiles = W.view(M, N // 128, 128)
    m = tiles.abs().amax(dim=2)
    codes = torch.zeros_like(m, dtype=torch.int64)
    scales = torch.zeros_like(m, dtype=torch.float64)
    nonzero = m > 0
    e = torch.ceil(torch.log2(m[nonzero].to(torch.float64)))
    e = torch.clamp(e, -126, 128).to(torch.int64)
    c = torch.clamp(e + 127, 1, 255)
    codes[nonzero] = c
    scales[nonzero] = torch.exp2(c.to(torch.float64) - 127.0)
    scale_codes = codes.to(torch.uint8)
    safe_scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    normalized = tiles.to(torch.float64) / safe_scales.unsqueeze(2)
    normalized = torch.where(scales.unsqueeze(2) > 0, normalized, torch.zeros_like(normalized))
    return scale_codes, normalized.to(W.dtype).view(M, N)
