import math

import torch


def _alpha_sigma(alpha_bar):
    ab = float(alpha_bar)
    if not (0.0 < ab < 1.0):
        raise ValueError("alpha_bar 必须严格落在 (0, 1) 内")
    alpha = math.sqrt(ab)
    sigma = math.sqrt(1.0 - ab)
    return alpha, sigma, math.log(alpha / sigma)


def solve(x, x0_pred, alpha_bar_t, alpha_bar_s0, order=1, x0_pred_prev=None, alpha_bar_s1=None):
    if not torch.is_tensor(x) or not torch.is_tensor(x0_pred):
        raise ValueError("x 与 x0_pred 必须是 torch.Tensor")
    if x.shape != x0_pred.shape:
        raise ValueError(f"形状不一致：{tuple(x.shape)} vs {tuple(x0_pred.shape)}")
    if order not in (1, 2):
        raise ValueError("order 只能是 1 或 2")

    alpha_t, sigma_t, lam_t = _alpha_sigma(alpha_bar_t)
    alpha_s0, sigma_s0, lam_s0 = _alpha_sigma(alpha_bar_s0)

    h = lam_t - lam_s0
    coef = -alpha_t * (math.exp(-h) - 1.0)  # 恒为正：从 s0 走向更干净的 t

    d0 = x0_pred.to(torch.float64)
    total = d0
    if order == 2:
        if x0_pred_prev is None or alpha_bar_s1 is None:
            raise ValueError("order=2 需要 x0_pred_prev 与 alpha_bar_s1")
        if not torch.is_tensor(x0_pred_prev):
            raise ValueError("x0_pred_prev 必须是 torch.Tensor")
        if x0_pred_prev.shape != d0.shape:
            raise ValueError("x0_pred_prev 形状必须与 x0_pred 一致")
        _, _, lam_s1 = _alpha_sigma(alpha_bar_s1)
        r0 = lam_s0 - lam_s1
        if r0 <= 0:
            raise ValueError("要求 λ_s0 > λ_s1（s1 必须是更噪的上一级）")
        total = d0 + 0.5 * (d0 - x0_pred_prev.to(torch.float64)) / r0

    out = (sigma_t / sigma_s0) * x.to(torch.float64) + coef * total
    return out.to(torch.float32)
