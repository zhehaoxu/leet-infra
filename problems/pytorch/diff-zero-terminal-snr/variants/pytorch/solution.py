import math

import torch


def solve(betas):
    if not torch.is_tensor(betas) or betas.dim() != 1:
        raise ValueError("betas 必须是一维 torch.Tensor")
    if betas.numel() == 0:
        raise ValueError("betas 不能为空")

    b = betas.to(torch.float64)
    if not torch.isfinite(b).all():
        raise ValueError("betas 里不能有 nan / inf")
    if bool((b < 0.0).any()) or bool((b >= 1.0).any()):
        raise ValueError("beta 必须落在 [0, 1) 内")

    alpha_bars = torch.cumprod(1.0 - b, dim=0)
    last = float(alpha_bars[-1])
    if last >= 1.0:
        raise ValueError("整条 β 全为 0 时 ᾱ_T = 1，无法重标定到零终端 SNR")

    alpha_bars = ((alpha_bars - last) / (1.0 - last)).clamp(0.0, 1.0)

    log_snrs = torch.empty_like(alpha_bars)
    angles = torch.empty_like(alpha_bars)
    for i, ab in enumerate(alpha_bars.tolist()):
        if ab <= 0.0:
            log_snrs[i] = float("-inf")
            angles[i] = math.pi / 2.0
        elif ab >= 1.0:
            log_snrs[i] = float("inf")
            angles[i] = 0.0
        else:
            log_snrs[i] = math.log(ab / (1.0 - ab))
            angles[i] = math.atan2(math.sqrt(1.0 - ab), math.sqrt(ab))

    return (
        alpha_bars.to(torch.float32),
        log_snrs.to(torch.float32),
        angles.to(torch.float32),
    )
