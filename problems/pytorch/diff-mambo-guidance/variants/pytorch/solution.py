import math

import torch


def _scalar(name, value, low=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是实数")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} 必须是有限实数")
    if low is not None and value < low:
        raise ValueError(f"{name} 不能小于 {low}")
    return value


def _tensor(name, value, shape=None):
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} 必须是 torch.Tensor")
    if value.dim() < 2:
        raise ValueError(f"{name} 至少要有两维：第 0 维 batch，其余维度参与归约")
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} 的形状必须与 eps_cond 一致")
    return value


def solve(eps_cond, eps_uncond, guidance_scale, alpha=8.0):
    """MAMBO-G：按「引导更新」与「无条件预测」的幅值比压回引导强度。

    eps_cond / eps_uncond:同形状预测，第 0 维是 batch；
    guidance_scale:w，基础引导强度；
    alpha:抑制强度 λ ≥ 0，越大抑制越激进。

    逐样本（除第 0 维外全部维度）：
      delta = eps_cond - eps_uncond
      r     = ‖delta‖₂ / ‖eps_uncond‖₂
      w'    = 1 + (w - 1) · exp(-λ·r)
      out   = eps_uncond + w' · delta

    ‖eps_uncond‖₂ = 0 时把 r 视为 +∞（即 w' = 1，输出退化为 eps_cond），不得出现 NaN。
    返回 float32 张量，形状与 eps_cond 相同。
    """
    cond = _tensor("eps_cond", eps_cond)
    uncond = _tensor("eps_uncond", eps_uncond, tuple(cond.shape))

    scale = _scalar("guidance_scale", guidance_scale, 0.0)
    strength = _scalar("alpha", alpha, 0.0)

    red = tuple(range(1, cond.dim()))
    c = cond.to(torch.float64)
    u = uncond.to(torch.float64)
    diff = c - u

    norm_diff = torch.linalg.vector_norm(diff, ord=2, dim=red, keepdim=True)
    norm_uncond = torch.linalg.vector_norm(u, ord=2, dim=red, keepdim=True)

    decay = torch.zeros_like(norm_diff)
    positive = norm_uncond > 0.0
    ratio = norm_diff[positive] / norm_uncond[positive]
    decay[positive] = torch.exp(-strength * ratio)

    scale_final = 1.0 + (scale - 1.0) * decay
    pred = u + scale_final * diff

    return pred.to(torch.float32)
