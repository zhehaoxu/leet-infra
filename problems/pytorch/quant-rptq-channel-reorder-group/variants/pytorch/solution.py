import torch

CLIPMIN = 1e-4


def solve(xmax, xmin, W, num_groups, bits):
    """RPTQ：按通道量化跨度排序重排，再逐组做组内非对称量化。"""
    for name, val in (("num_groups", num_groups), ("bits", bits)):
        if isinstance(val, bool) or not isinstance(val, int):
            raise ValueError(f"{name} 必须是整数")
    if num_groups < 1:
        raise ValueError("num_groups 必须 >= 1")
    if not 2 <= bits <= 8:
        raise ValueError("bits 必须在 [2, 8]")
    if not isinstance(xmax, torch.Tensor) or not isinstance(xmin, torch.Tensor):
        raise ValueError("xmax / xmin 必须是张量")
    if xmax.dim() != 1 or xmax.shape != xmin.shape or xmax.numel() == 0:
        raise ValueError("xmax / xmin 必须等长一维且非空")
    if not xmax.is_floating_point() or not xmin.is_floating_point():
        raise ValueError("xmax / xmin 必须是浮点张量")
    if not isinstance(W, torch.Tensor) or W.dim() != 2 or not W.is_floating_point():
        raise ValueError("W 必须是二维浮点张量")
    C = xmax.numel()
    if W.shape[0] == 0 or W.shape[1] != C:
        raise ValueError("W 的列数必须等于通道数")
    if num_groups > C:
        raise ValueError("num_groups 不能超过通道数")
    if not bool(torch.isfinite(xmax).all() and torch.isfinite(xmin).all()):
        raise ValueError("xmax / xmin 必须有限")
    if bool((xmin > xmax).any()):
        raise ValueError("必须满足 xmin <= xmax")
    if not bool(torch.isfinite(W).all()):
        raise ValueError("W 必须有限")

    span = (xmax.to(torch.float64) - xmin.to(torch.float64))
    perm = torch.argsort(span, stable=True)
    sizes = [C // num_groups + (1 if j < C % num_groups else 0) for j in range(num_groups)]
    counts = torch.tensor(sizes, dtype=torch.int64)

    xmax64 = xmax.to(torch.float64)
    xmin64 = xmin.to(torch.float64)
    qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    scales = torch.empty(num_groups, dtype=torch.float64)
    zeros = torch.empty(num_groups, dtype=torch.float64)
    dequant = torch.empty_like(W, dtype=torch.float64)

    start = 0
    for j in range(num_groups):
        idx = perm[start:start + sizes[j]]
        start += sizes[j]
        gmax = xmax64[idx].amax()
        gmin = xmin64[idx].amin()
        scale = ((gmax - gmin) * (2.0 ** -bits)).clamp(min=CLIPMIN, max=1e4)
        zero = ((gmax + gmin) * (-0.5 / scale)).clamp(min=-1e4, max=1e4).round()
        scales[j] = scale
        zeros[j] = zero
        w = W[:, idx].to(torch.float64)
        q = (w / scale).round().add(zero).clamp(qmin, qmax).sub(zero)
        dequant[:, idx] = q * scale

    return {
        "perm": perm.to(torch.int64),
        "counts": counts,
        "scales": scales.to(torch.float32),
        "zero_points": zeros.to(torch.int64),
        "dequant": dequant.to(torch.float32),
    }
