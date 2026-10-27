import math

import torch


def _check_schedule(alpha_bars):
    if not torch.is_tensor(alpha_bars) or alpha_bars.dim() != 1:
        raise ValueError("alpha_bars 必须是一维 torch.Tensor")
    if alpha_bars.numel() < 2:
        raise ValueError("alpha_bars 至少要有两个元素（干净 → 噪声）")
    ab = alpha_bars.to(torch.float64)
    if not torch.isfinite(ab).all():
        raise ValueError("alpha_bars 里不能有 nan / inf")
    if bool((ab <= 0.0).any()) or bool((ab > 1.0).any()):
        raise ValueError("alpha_bars 必须落在 (0, 1] 内")
    if not bool((ab[:-1] > ab[1:]).all()):
        raise ValueError("alpha_bars 必须严格递减（按噪声级升序排列）")
    return ab


def solve(x0, noise, alpha_bars, strength):
    if not torch.is_tensor(x0) or not torch.is_tensor(noise):
        raise ValueError("x0 与 noise 必须是 torch.Tensor")
    if x0.shape != noise.shape:
        raise ValueError(f"形状不一致：{tuple(x0.shape)} vs {tuple(noise.shape)}")

    s = float(strength)
    if not math.isfinite(s) or not (0.0 <= s <= 1.0):
        raise ValueError("strength 必须落在 [0, 1] 内")

    ab = _check_schedule(alpha_bars)
    last = ab.numel() - 1
    idx = int(math.floor(s * last))
    idx = min(max(idx, 0), last)

    cur = float(ab[idx])
    alpha = math.sqrt(cur)
    sigma = math.sqrt(1.0 - cur)
    out = alpha * x0.to(torch.float64) + sigma * noise.to(torch.float64)
    return out.to(torch.float32), idx
