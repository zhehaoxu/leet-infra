import math

import torch


def _positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} 必须是整数")
    if value < 0:
        raise ValueError(f"{name} 必须非负")
    return value


def solve(eps_cond, eps_uncond, guidance_scale, step, zero_init_steps=1, eps=1e-8):
    if not torch.is_tensor(eps_cond) or not torch.is_tensor(eps_uncond):
        raise ValueError("eps_cond 与 eps_uncond 必须是 torch.Tensor")
    if eps_cond.shape != eps_uncond.shape:
        raise ValueError(f"形状不一致：{tuple(eps_cond.shape)} vs {tuple(eps_uncond.shape)}")
    if eps_cond.dim() < 2:
        raise ValueError("至少需要二维输入（batch × 特征）")

    scale = float(guidance_scale)
    if not math.isfinite(scale) or scale < 0.0:
        raise ValueError("guidance_scale 必须是非负有限实数")
    eps_value = float(eps)
    if not math.isfinite(eps_value) or eps_value <= 0.0:
        raise ValueError("eps 必须是正有限实数")
    current = _positive_int("step", step)
    warmup = _positive_int("zero_init_steps", zero_init_steps)

    if current < warmup:
        return torch.zeros_like(eps_cond, dtype=torch.float32)

    cond = eps_cond.to(torch.float64)
    uncond = eps_uncond.to(torch.float64)
    red = tuple(range(1, cond.dim()))

    dot = torch.sum(cond * uncond, dim=red, keepdim=True)
    squared = torch.sum(uncond * uncond, dim=red, keepdim=True) + eps_value
    # 最小二乘意义下把 uncond 对齐到 cond 的最优尺度
    alpha = dot / squared
    scaled_uncond = alpha * uncond
    pred = scaled_uncond + scale * (cond - scaled_uncond)
    return pred.to(torch.float32)
