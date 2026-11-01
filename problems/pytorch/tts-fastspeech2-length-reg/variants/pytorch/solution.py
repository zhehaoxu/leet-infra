import math


def _linspace(lo, hi, num):
    # torch.linspace：端点精确，中间等步长
    if num == 1:
        return [lo]
    return [lo + i * (hi - lo) / (num - 1) for i in range(num)]


def solve(log_durations, duration_target, d_control,
          expand_features, feat_value, p_control,
          feat_min, feat_max, n_bins, log_scale):
    T = len(log_durations)
    assert len(expand_features) == T
    if duration_target is not None:
        assert len(duration_target) == T
        duration_rounded = list(duration_target)
    else:
        duration_rounded = []
        for logd in log_durations:
            d = math.exp(logd) - 1.0
            r = max(round(d) * d_control, 0)
            duration_rounded.append(r)

    expand_indices = []
    for i, d in enumerate(duration_rounded):
        n = max(int(d), 0)
        expand_indices.extend([i] * n)

    if log_scale:
        lin = _linspace(math.log(feat_min), math.log(feat_max), n_bins - 1)
        bins = [math.exp(x) for x in lin]
    else:
        bins = _linspace(feat_min, feat_max, n_bins - 1)

    value = feat_value * p_control
    lo, hi = 0, len(bins)
    while lo < hi:
        mid = (lo + hi) // 2
        if bins[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    feat_bucket = lo

    return {
        "duration_rounded": duration_rounded,
        "mel_len": len(expand_indices),
        "expand_indices": expand_indices,
        "feat_bucket": feat_bucket,
    }
