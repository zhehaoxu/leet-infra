import math

import torch

_PREDICTION_TYPES = ("epsilon", "sample", "v_prediction")


def _as_int(name, value, low, high):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} 必须是 Python 整数")
    if value < low or value > high:
        raise ValueError(f"{name} 必须落在 [{low}, {high}] 内")
    return value


def _as_float(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是实数")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} 必须是有限值")
    return value


def solve(
    alphas_cumprod,
    timestep,
    sample,
    model_output,
    num_inference_steps,
    prediction_type="epsilon",
    clip_sample=True,
    clip_sample_range=1.0,
    set_alpha_to_one=True,
):
    if not torch.is_tensor(alphas_cumprod) or alphas_cumprod.dim() != 1:
        raise ValueError("alphas_cumprod 必须是一维张量")
    if alphas_cumprod.dtype not in (torch.float32, torch.float64):
        raise ValueError("alphas_cumprod 必须是 float32 / float64")
    if alphas_cumprod.numel() < 1:
        raise ValueError("alphas_cumprod 不能为空")
    ac = alphas_cumprod.to(torch.float64)
    if not bool(torch.isfinite(ac).all()):
        raise ValueError("alphas_cumprod 不能含 nan / inf")
    if not bool(((ac > 0.0) & (ac <= 1.0)).all()):
        raise ValueError("alphas_cumprod 的每个元素必须落在 (0, 1] 内")
    num_train = ac.numel()

    timestep = _as_int("timestep", timestep, 0, num_train - 1)
    num_inference_steps = _as_int("num_inference_steps", num_inference_steps, 1, num_train)
    clip_sample_range = _as_float("clip_sample_range", clip_sample_range)
    if clip_sample_range <= 0.0:
        raise ValueError("clip_sample_range 必须为正")
    if not isinstance(clip_sample, bool):
        raise ValueError("clip_sample 必须是布尔值")
    if not isinstance(set_alpha_to_one, bool):
        raise ValueError("set_alpha_to_one 必须是布尔值")
    if prediction_type not in _PREDICTION_TYPES:
        raise ValueError("prediction_type 只能是 epsilon / sample / v_prediction")

    for name, value in (("sample", sample), ("model_output", model_output)):
        if not torch.is_tensor(value) or value.dtype not in (torch.float32, torch.float64):
            raise ValueError(f"{name} 必须是 float32 / float64 张量")
        if value.numel() < 1:
            raise ValueError(f"{name} 不能为空")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} 不能含 nan / inf")
    if sample.shape != model_output.shape:
        raise ValueError("sample 与 model_output 必须同形状")

    prev_timestep = timestep
    t = min(timestep - num_train // num_inference_steps, num_train - 1)
    if t >= 0:
        alpha_prod_t = ac[t]
    else:
        alpha_prod_t = torch.tensor(1.0, dtype=torch.float64) if set_alpha_to_one else ac[0]
    alpha_prod_t_prev = ac[prev_timestep]
    beta_prod_t = 1.0 - alpha_prod_t

    xs = sample.to(torch.float64)
    mo = model_output.to(torch.float64)
    if prediction_type == "epsilon":
        pred_original = (xs - beta_prod_t ** 0.5 * mo) / alpha_prod_t ** 0.5
        pred_epsilon = mo
    elif prediction_type == "sample":
        if float(beta_prod_t) <= 0.0:
            raise ValueError("sample 预测类型要求 beta_prod_t > 0")
        pred_original = mo
        pred_epsilon = (xs - alpha_prod_t ** 0.5 * pred_original) / beta_prod_t ** 0.5
    else:
        pred_original = alpha_prod_t ** 0.5 * xs - beta_prod_t ** 0.5 * mo
        pred_epsilon = alpha_prod_t ** 0.5 * mo + beta_prod_t ** 0.5 * xs

    if clip_sample:
        pred_original = pred_original.clamp(-clip_sample_range, clip_sample_range)

    pred_sample_direction = (1.0 - alpha_prod_t_prev) ** 0.5 * pred_epsilon
    prev_sample = alpha_prod_t_prev ** 0.5 * pred_original + pred_sample_direction
    return prev_sample.to(torch.float32), pred_original.to(torch.float32)
