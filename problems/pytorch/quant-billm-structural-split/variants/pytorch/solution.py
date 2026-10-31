import torch


def _high_order_residual(x, mask, order):
    out = torch.zeros_like(x)
    count = mask.sum(dim=1, keepdim=True).to(x.dtype).clamp_min(1.0)
    zero = torch.zeros_like(x)
    for _ in range(order):
        residual = torch.where(mask, x - out, zero)
        mean = residual.sum(dim=1, keepdim=True) / count
        centered = torch.where(mask, residual - mean, zero)
        scale = centered.abs().sum(dim=1, keepdim=True) / count
        binary = torch.sign(centered) * scale + mean
        out = out + torch.where(mask, binary, zero)
    return out


def _quantiles(values, qs):
    n = values.numel()
    pos = qs * (n - 1)
    lo = torch.floor(pos).to(torch.int64)
    hi = torch.clamp(lo + 1, max=n - 1)
    frac = pos - lo.to(torch.float64)
    return values[lo] + (values[hi] - values[lo]) * frac


def solve(W, up_lim):
    """BiLLM：按列分组做结构化拆分，组内用逐行残差二值重建。"""
    if not isinstance(W, torch.Tensor):
        raise ValueError("W 必须是 torch.Tensor")
    if W.dim() != 2:
        raise ValueError("W 必须是二维张量")
    if not W.is_floating_point():
        raise ValueError("W 必须是浮点类型")
    if W.numel() == 0:
        raise ValueError("W 不能为空")
    if not bool(torch.isfinite(W).all()):
        raise ValueError("W 必须全部有限")
    if isinstance(up_lim, bool) or not isinstance(up_lim, int):
        raise ValueError("up_lim 必须是整数")
    cols = W.shape[1]
    if not 2 <= up_lim <= cols:
        raise ValueError("up_lim 必须满足 2 <= up_lim <= 列数")

    x = W.to(torch.float64)
    rows = x.shape[0]
    abs_x = x.abs()
    counts = abs_x.sum(dim=0)
    order_idx = torch.argsort(counts, descending=True, stable=True)
    candidates = order_idx[:up_lim]

    best_i = 1
    best_err = float("inf")
    for i in range(1, up_lim):
        mask3 = torch.zeros((rows, cols), dtype=torch.bool)
        mask3[:, candidates[:i]] = True
        rec = _high_order_residual(x, ~mask3, 2) + _high_order_residual(x, mask3, 2)
        err = float(((x - rec) ** 2).mean())
        if err < best_err:
            best_err = err
            best_i = i

    selected = order_idx[:best_i]
    salient = torch.sort(selected).values
    mask3 = torch.zeros((rows, cols), dtype=torch.bool)
    mask3[:, selected] = True
    group3 = _high_order_residual(x, mask3, 2)

    zero = torch.zeros_like(x)
    flat = torch.where(mask3, zero, x).abs().flatten()
    values = torch.sort(flat).values
    grid = torch.linspace(0.10, 0.90, 81, dtype=torch.float64)
    splits = _quantiles(values, grid)

    best_split = None
    best_rec = None
    best_err = float("inf")
    for split in splits.tolist():
        mask2 = (~mask3) & (abs_x >= split)
        mask1 = (~mask3) & ~mask2
        rec = (
            _high_order_residual(x, mask1, 1)
            + _high_order_residual(x, mask2, 1)
            + group3
        )
        err = float(((x - rec) ** 2).mean())
        if err < best_err:
            best_err = err
            best_split = split
            best_rec = rec
            best_mask1 = mask1
            best_mask2 = mask2

    return {
        "salient_cols": salient.to(torch.int64),
        "split": torch.tensor(best_split, dtype=torch.float32),
        "mask1": best_mask1,
        "mask2": best_mask2,
        "mask3": mask3,
        "reconstructed": best_rec.to(torch.float32),
    }
