import math

import torch


def _mu_from_seq(seq_len, base_shift, max_shift, base_seq_len, max_seq_len):
    if isinstance(seq_len, bool) or not isinstance(seq_len, int):
        raise ValueError("seq_len 必须是整数")
    if seq_len < 0:
        raise ValueError("seq_len 不能为负")
    if float(max_seq_len) <= float(base_seq_len):
        raise ValueError("要求 max_seq_len > base_seq_len")
    slope = (float(max_shift) - float(base_shift)) / (float(max_seq_len) - float(base_seq_len))
    intercept = float(base_shift) - slope * float(base_seq_len)
    return slope * float(seq_len) + intercept


def solve(
    times,
    shift=None,
    mu=None,
    seq_len=None,
    base_shift=0.5,
    max_shift=1.15,
    base_seq_len=256,
    max_seq_len=4096,
):
    if not torch.is_tensor(times) or times.dim() != 1:
        raise ValueError("times 必须是一维 torch.Tensor")
    if sum(x is not None for x in (shift, mu, seq_len)) != 1:
        raise ValueError("shift / mu / seq_len 必须恰给一个")

    if seq_len is not None:
        mu = _mu_from_seq(seq_len, base_shift, max_shift, base_seq_len, max_seq_len)
    if mu is not None:
        mu = float(mu)
        if not math.isfinite(mu):
            raise ValueError("mu 必须是有限实数")
        s = math.exp(mu)
    else:
        s = float(shift)
    if not math.isfinite(s) or s <= 0.0:
        raise ValueError("shift 必须为正的有限实数")

    t = times.to(torch.float64)
    if not torch.isfinite(t).all():
        raise ValueError("times 里不能有 nan / inf")
    if bool((t < 0.0).any()) or bool((t > 1.0).any()):
        raise ValueError("times 必须落在 [0, 1] 内")

    shifted = s * t / (1.0 + (s - 1.0) * t)
    return shifted.to(torch.float32), s
