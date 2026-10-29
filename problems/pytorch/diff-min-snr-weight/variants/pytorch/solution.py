import math

import torch


def solve(alphas_cumprod, timesteps, gamma):
    if not torch.is_tensor(alphas_cumprod):
        raise ValueError("alphas_cumprod 必须是 torch 张量")
    if alphas_cumprod.ndim != 1:
        raise ValueError("alphas_cumprod 必须是一维张量")
    if not alphas_cumprod.is_floating_point():
        raise ValueError("alphas_cumprod 必须是浮点张量")

    ac = alphas_cumprod.to(torch.float64)
    if not bool(torch.isfinite(ac).all()):
        raise ValueError("alphas_cumprod 不能含 nan / inf")
    if bool(((ac <= 0) | (ac >= 1)).any()):
        raise ValueError("alphas_cumprod 必须严格落在 (0, 1) 内")

    if not torch.is_tensor(timesteps):
        raise ValueError("timesteps 必须是 torch 张量")
    if timesteps.ndim != 1:
        raise ValueError("timesteps 必须是一维张量")
    if timesteps.dtype not in (torch.int32, torch.int64):
        raise ValueError("timesteps 必须是 int32 或 int64 张量")
    if timesteps.numel() > 0:
        if int(timesteps.min()) < 0 or int(timesteps.max()) >= ac.numel():
            raise ValueError("timesteps 越界")

    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)):
        raise ValueError("gamma 必须是实数")
    g = float(gamma)
    if not math.isfinite(g) or g <= 0:
        raise ValueError("gamma 必须是正的有限实数")

    idx = timesteps.to(torch.int64)
    alpha = torch.sqrt(ac)[idx]
    sigma = torch.sqrt(1.0 - ac)[idx]
    snr = (alpha / sigma) ** 2
    weight = torch.minimum(snr, torch.full_like(snr, g)) / snr

    return snr.to(torch.float32), weight.to(torch.float32)
