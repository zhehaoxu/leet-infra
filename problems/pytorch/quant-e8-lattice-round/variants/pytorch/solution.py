import torch


def _d8_nearest(x):
    """x [N, 8] float64 -> 最近的 D8 点（坐标和为偶数）。"""
    y = x.round()
    diff = x - y
    need = (y.sum(dim=1) % 2 != 0)
    absdiff = diff.abs()
    best = absdiff.amax(dim=1, keepdim=True)
    idx = torch.where(absdiff == best, torch.arange(8), 8).amin(dim=1)
    step = torch.sign(diff.gather(1, idx.unsqueeze(1)))
    step = torch.where(step == 0, torch.ones_like(step), step)
    rows = torch.arange(x.shape[0])
    y = y.clone()
    y[rows[need], idx[need]] += step[need, 0]
    return y


def solve(X):
    """E8 = D8 ∪ (D8 + 1/2)：分别在两个陪集里找最近点，取误差更小者。"""
    if not isinstance(X, torch.Tensor) or X.dim() != 2 or not X.is_floating_point():
        raise ValueError("X 必须是二维浮点张量")
    if X.shape[0] == 0 or X.shape[1] != 8:
        raise ValueError("X 的形状必须是 [N, 8] 且 N > 0")
    if not bool(torch.isfinite(X).all()):
        raise ValueError("X 必须有限")

    x = X.to(torch.float64)
    y0 = _d8_nearest(x)
    y1 = _d8_nearest(x - 0.5) + 0.5

    e0 = ((x - y0) ** 2).sum(dim=1)
    e1 = ((x - y1) ** 2).sum(dim=1)
    use_coset1 = e1 < e0

    rounded = torch.where(use_coset1.unsqueeze(1), y1, y0)
    sse = torch.where(use_coset1, e1, e0)
    return {
        "rounded": rounded.to(torch.float32),
        "coset": use_coset1.to(torch.int64),
        "sse": sse.to(torch.float32),
    }
