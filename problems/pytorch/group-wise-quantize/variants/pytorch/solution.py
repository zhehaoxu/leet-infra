import torch


def solve(W: torch.Tensor, group_size: int, bits: int = 8) -> torch.Tensor:
    qmax = 2 ** (bits - 1) - 1
    shape = W.shape
    n = shape[-1]
    flat = W.clone().reshape(-1, n)
    rows = flat.shape[0]
    for r in range(rows):
        for start in range(0, n, group_size):
            g = flat[r, start:start + group_size]  # 最后一组自动为短尾
            amax = g.abs().max()
            if float(amax) == 0.0:
                continue  # 全零组原样保留 0
            scale = amax / qmax
            q = torch.clamp(torch.round(g / scale), -qmax, qmax)
            flat[r, start:start + group_size] = q * scale
    return flat.reshape(shape)
