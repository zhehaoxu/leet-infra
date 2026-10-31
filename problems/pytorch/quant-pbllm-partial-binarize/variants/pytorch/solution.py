import torch


def solve(W, low_frac, high_bits):
    """PB-LLM：幅度最小的 low_frac 比例权重走二值（xnor）通路，其余走高位非对称量化。"""
    if isinstance(high_bits, bool) or not isinstance(high_bits, int) or not 2 <= high_bits <= 8:
        raise ValueError("high_bits 必须是 [2, 8] 的整数")
    if isinstance(low_frac, bool) or not isinstance(low_frac, (int, float)):
        raise ValueError("low_frac 必须是数值")
    low_frac = float(low_frac)
    if not (0.0 <= low_frac < 1.0):
        raise ValueError("low_frac 必须满足 0 <= low_frac < 1")
    if not isinstance(W, torch.Tensor) or W.dim() != 2 or not W.is_floating_point():
        raise ValueError("W 必须是二维浮点张量")
    if W.numel() == 0:
        raise ValueError("W 不能为空")
    if not bool(torch.isfinite(W).all()):
        raise ValueError("W 必须有限")

    x = W.to(torch.float64)
    maxq = 2 ** high_bits - 1

    thresh = x.abs().flatten().sort().values[int(x.numel() * low_frac)]
    mask = x.abs() <= thresh

    masked = x * mask
    low_mean = masked.mean(dim=1)
    low_scale = (masked - low_mean.unsqueeze(1)).abs().mean(dim=1)

    zero = torch.zeros((), dtype=torch.float64)
    xmin = torch.minimum(x.amin(dim=1), zero)
    xmax = torch.maximum(x.amax(dim=1), zero)
    both_zero = (xmin == 0) & (xmax == 0)
    xmin = torch.where(both_zero, -torch.ones_like(xmin), xmin)
    xmax = torch.where(both_zero, torch.ones_like(xmax), xmax)
    high_scale = (xmax - xmin) / maxq
    high_zero = (-xmin / high_scale).round().to(torch.int64)

    hz = high_zero.to(torch.float64).unsqueeze(1)
    hs = high_scale.unsqueeze(1)
    q = (x / hs).round().add(hz).clamp(0, maxq)
    high_dequant = (q - hz) * hs

    centered = x - low_mean.unsqueeze(1)
    low_dequant = torch.sign(centered) * low_scale.unsqueeze(1) + low_mean.unsqueeze(1)
    dequant = torch.where(mask, low_dequant, high_dequant)

    return {
        "mask": mask,
        "low_mean": low_mean.to(torch.float32),
        "low_scale": low_scale.to(torch.float32),
        "high_scale": high_scale.to(torch.float32),
        "high_zero": high_zero,
        "dequant": dequant.to(torch.float32),
    }
