import math

import torch


def _tensor(name, value, shape=None):
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} 必须是 torch.Tensor")
    if value.dim() < 1:
        raise ValueError(f"{name} 至少是一维张量（第 0 维为 batch）")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} 含 NaN 或 inf")
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} 形状应为 {tuple(shape)}，实际 {tuple(value.shape)}")
    return value.to(torch.float64)


def _int(name, value, low=0):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} 必须是 Python int")
    if value < low:
        raise ValueError(f"{name} 不能小于 {low}")
    return value


def _float(name, value, low=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是实数")
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{name} 必须是有限值")
    if low is not None and v < low:
        raise ValueError(f"{name} 不能小于 {low}")
    return v


def solve(sample, model_output, original_image, mask, noise, betas, undo_noise, timestep,
          num_inference_steps, eta=0.0, clip_sample=True):
    """RePaint 的掩码拼接步与 undo_step 前向回跳（diffusers `RePaintScheduler`）。"""
    x = _tensor("sample", sample)
    shape = tuple(x.shape)
    eps = _tensor("model_output", model_output, shape)
    orig = _tensor("original_image", original_image, shape)
    m = _tensor("mask", mask, shape)
    z = _tensor("noise", noise, shape)
    if bool((m < 0).any()) or bool((m > 1).any()):
        raise ValueError("mask 的取值必须落在 [0, 1]：1 表示保留原图，0 表示待补区域")

    if not isinstance(betas, torch.Tensor):
        raise ValueError("betas 必须是 torch.Tensor")
    if betas.dim() != 1:
        raise ValueError("betas 必须是一维张量")
    if not torch.isfinite(betas).all():
        raise ValueError("betas 含 NaN 或 inf")
    b = betas.to(torch.float64)
    if bool((b < 0).any()) or bool((b >= 1).any()):
        raise ValueError("betas 必须全部落在 [0, 1) 内")
    abar = torch.cumprod(1.0 - b, dim=0)
    if bool((abar <= 0).any()):
        raise ValueError("betas 使累积乘积 ᾱ 触到 0，本题的 step 无法在该表上求值")
    T = int(b.shape[0])

    n_in = _int("num_inference_steps", num_inference_steps, low=1)
    if T % n_in != 0:
        raise ValueError("betas 长度必须能被 num_inference_steps 整除（上游用整除算子步长 n）")
    n = T // n_in

    t = _int("timestep", timestep)
    if t >= T:
        raise ValueError(f"timestep 不能大于等于 len(betas)={T}")
    if t + n > T:
        raise ValueError(f"timestep + n 越界：undo_step 会索引到 betas[{T}] 之后")

    eta_v = _float("eta", eta, low=0.0)
    if not isinstance(clip_sample, bool):
        raise ValueError("clip_sample 必须是 bool")

    if not isinstance(undo_noise, torch.Tensor):
        raise ValueError("undo_noise 必须是 torch.Tensor")
    if tuple(undo_noise.shape) != (n,) + shape:
        raise ValueError(
            f"undo_noise 形状应为 {(n,) + shape}（回跳的 n={n} 个子步各一张噪声），实际 {tuple(undo_noise.shape)}"
        )
    un = _tensor("undo_noise", undo_noise)

    alpha_prod_t = abar[t]
    prev_t = t - n
    alpha_prod_t_prev = abar[prev_t] if prev_t >= 0 else torch.ones((), dtype=torch.float64)
    beta_prod_t = 1 - alpha_prod_t

    pred_original_sample = (x - beta_prod_t**0.5 * eps) / alpha_prod_t**0.5
    if clip_sample:
        pred_original_sample = torch.clamp(pred_original_sample, -1.0, 1.0)

    variance = ((1 - alpha_prod_t_prev) / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
    std_dev_t = eta_v * variance**0.5

    added = 0.0
    if t > 0 and eta_v > 0:
        added = std_dev_t * z

    direction_arg = 1 - alpha_prod_t_prev - std_dev_t**2
    if bool((direction_arg < 0).any()):
        raise ValueError("eta 过大使 1 - ᾱ_prev - (eta·sigma_t)^2 为负，方向项的平方根无定义")
    pred_sample_direction = direction_arg**0.5 * eps

    prev_unknown_part = alpha_prod_t_prev**0.5 * pred_original_sample + pred_sample_direction + added
    prev_known_part = alpha_prod_t_prev**0.5 * orig + (1 - alpha_prod_t_prev) ** 0.5 * z
    prev_sample = m * prev_known_part + (1.0 - m) * prev_unknown_part

    undone = x
    for i in range(n):
        beta_i = b[t + i]
        undone = (1 - beta_i) ** 0.5 * undone + beta_i**0.5 * un[i]

    return (prev_sample.to(torch.float32), pred_original_sample.to(torch.float32), undone.to(torch.float32))
