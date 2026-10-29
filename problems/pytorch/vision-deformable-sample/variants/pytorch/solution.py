import torch


def solve(feature, points, offsets, weights=None):
    if not isinstance(feature, torch.Tensor):
        raise ValueError("feature 必须是 torch.Tensor")
    if feature.dim() != 3:
        raise ValueError(f"feature 必须是 (C, H, W) 三维，got {tuple(feature.shape)}")
    c, h, w = (int(v) for v in feature.shape)
    if c < 1 or h < 1 or w < 1:
        raise ValueError(f"feature 的 C/H/W 都必须 >= 1，got {(c, h, w)}")
    if feature.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"feature dtype 必须是 float32/float64，got {feature.dtype}")

    if not isinstance(points, torch.Tensor) or not isinstance(offsets, torch.Tensor):
        raise ValueError("points 与 offsets 都必须是 torch.Tensor")
    if points.dim() != 3 or points.shape[-1] != 2:
        raise ValueError(f"points 必须是 (N, K, 2) 三维，got {tuple(points.shape)}")
    if tuple(points.shape) != tuple(offsets.shape):
        raise ValueError(
            f"points 与 offsets 形状必须一致，got {tuple(points.shape)} vs {tuple(offsets.shape)}")
    if points.dtype != feature.dtype:
        raise ValueError(f"points dtype 必须与 feature 相同，got {points.dtype}/{feature.dtype}")
    if offsets.dtype != feature.dtype:
        raise ValueError(f"offsets dtype 必须与 feature 相同，got {offsets.dtype}/{feature.dtype}")

    n, k = int(points.shape[0]), int(points.shape[1])
    if k < 1:
        raise ValueError(f"K 必须 >= 1，got {k}")
    if not bool(torch.isfinite(points).all()):
        raise ValueError("points 含 NaN 或 Inf")
    if not bool(torch.isfinite(offsets).all()):
        raise ValueError("offsets 含 NaN 或 Inf")

    if weights is not None:
        if not isinstance(weights, torch.Tensor):
            raise ValueError("weights 必须是 torch.Tensor 或 None")
        if tuple(weights.shape) != (n, k):
            raise ValueError(f"weights 形状必须是 ({n}, {k})，got {tuple(weights.shape)}")
        if weights.dtype != feature.dtype:
            raise ValueError(
                f"weights dtype 必须与 feature 相同，got {weights.dtype}/{feature.dtype}")
        if not bool(torch.isfinite(weights).all()):
            raise ValueError("weights 含 NaN 或 Inf")

    f64 = feature.to(torch.float64)
    pos = points.to(torch.float64) + offsets.to(torch.float64)
    x = pos[..., 0]
    y = pos[..., 1]
    x0 = torch.floor(x)
    y0 = torch.floor(y)
    fx = x - x0
    fy = y - y0

    out = torch.zeros((n, k, c), dtype=torch.float64)
    for dy, dx, wgt in (
        (0, 0, (1.0 - fx) * (1.0 - fy)),
        (0, 1, fx * (1.0 - fy)),
        (1, 0, (1.0 - fx) * fy),
        (1, 1, fx * fy),
    ):
        xi = x0 + dx
        yi = y0 + dy
        valid = (xi >= 0.0) & (xi <= w - 1.0) & (yi >= 0.0) & (yi <= h - 1.0)
        xi = xi.clamp(0.0, w - 1.0).to(torch.int64)
        yi = yi.clamp(0.0, h - 1.0).to(torch.int64)
        vals = f64[:, yi, xi].permute(1, 2, 0)
        out = out + (wgt * valid.to(torch.float64)).unsqueeze(-1) * vals

    if weights is None:
        return out.to(feature.dtype)
    return (out * weights.to(torch.float64).unsqueeze(-1)).sum(dim=1).to(feature.dtype)
