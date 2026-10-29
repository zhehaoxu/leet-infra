import torch

_DEFAULT_ASPECT_RATIOS = [
    (1, 1),
    (4, 5),
    (5, 4),
    (2, 3),
    (3, 2),
    (3, 4),
    (4, 3),
    (9, 16),
    (16, 9),
    (1, 2),
    (2, 1),
]


def _is_pos_int(v):
    return isinstance(v, int) and not isinstance(v, bool) and v > 0


def _build_buckets(resolution, divisibility, base_resolutions):
    if base_resolutions is None:
        ratios = list(_DEFAULT_ASPECT_RATIOS)
    else:
        if not isinstance(base_resolutions, (list, tuple)) or len(base_resolutions) == 0:
            raise ValueError("base_resolutions 必须是非空的 (h, w) 列表")
        ratios = []
        for pair in base_resolutions:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("base_resolutions 的每一项必须是 (h, w)")
            h, w = pair
            if not (_is_pos_int(h) and _is_pos_int(w)):
                raise ValueError("base_resolutions 的宽高必须是正整数")
            ratios.append((w, h))

    target = resolution * resolution
    buckets = []
    seen = set()
    for rw, rh in ratios:
        aspect = rw / rh
        h = (target / aspect) ** 0.5
        w = h * aspect
        h = max(divisibility, round(h / divisibility) * divisibility)
        w = max(divisibility, round(w / divisibility) * divisibility)
        if (h, w) not in seen:
            seen.add((h, w))
            buckets.append((h, w))
    return buckets


def solve(resolution, image_sizes, divisibility=16, base_resolutions=None):
    if not _is_pos_int(resolution):
        raise ValueError("resolution 必须是正整数")
    if not _is_pos_int(divisibility):
        raise ValueError("divisibility 必须是正整数")

    buckets = _build_buckets(resolution, divisibility, base_resolutions)

    if not torch.is_tensor(image_sizes):
        raise ValueError("image_sizes 必须是 torch 张量")
    if image_sizes.ndim != 2 or image_sizes.shape[1] != 2:
        raise ValueError("image_sizes 必须是形状 (N, 2) 的二维张量")
    if image_sizes.dtype not in (torch.int32, torch.int64):
        raise ValueError("image_sizes 必须是 int32 或 int64 张量")
    if image_sizes.numel() > 0 and int(image_sizes.min()) <= 0:
        raise ValueError("image_sizes 的宽高必须是正整数")

    assignments = []
    for h, w in image_sizes.tolist():
        best = 0
        best_metric = None
        for i, (bh, bw) in enumerate(buckets):
            metric = abs(h * bw - w * bh)
            if best_metric is None or metric <= best_metric:
                best_metric = metric
                best = i
        assignments.append(best)

    bucket_tensor = torch.tensor(buckets, dtype=torch.int64).reshape(len(buckets), 2)
    return bucket_tensor, torch.tensor(assignments, dtype=torch.int64)
