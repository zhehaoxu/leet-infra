import math

import torch

_PREDICTION_TYPES = ("epsilon", "sample", "v_prediction")
_SIGMA_DATA = 0.5


def _tensor(name, value):
    if not torch.is_tensor(value):
        raise ValueError(f"{name} 必须是 torch 张量")
    if not value.is_floating_point():
        raise ValueError(f"{name} 必须是浮点张量")
    if value.numel() == 0:
        raise ValueError(f"{name} 不能为空张量")
    return value.to(torch.float64)


def solve(
    sample,
    model_output,
    alphas_cumprod,
    timesteps,
    step_index,
    prediction_type="epsilon",
    clip_sample=True,
    clip_sample_range=1.0,
    timestep_scaling=10.0,
    noise=None,
):
    """LCM 单步：返回 (denoised, prev_sample, c_skip, c_out)。

    sample / model_output:同形状非空浮点张量；alphas_cumprod:一维浮点张量，长度 T，元素落在 (0, 1)；
    timesteps:一维整数张量（LCM 推理步的子采样），每个值都在 [0, T-1]；
    step_index:整数，落在 [0, len(timesteps)-1]。

    1) 边界条件系数（σ_d 上游写死 0.5）：
       s = timesteps[step_index] * timestep_scaling
       c_skip = 0.5**2 / (s**2 + 0.5**2)
       c_out  = s / (s**2 + 0.5**2) ** 0.5      （注意没有乘 σ_d）

    2) x0（用 alphas_cumprod 的平方根）：
       epsilon      -> x0 = (sample - sqrt(1 - ᾱ_t) * model_output) / sqrt(ᾱ_t)
       sample       -> x0 = model_output
       v_prediction -> x0 = sqrt(ᾱ_t) * sample - sqrt(1 - ᾱ_t) * model_output
       clip_sample=True 时再把 x0 钳到 [-clip_sample_range, clip_sample_range]。

    3) 边界条件混合：denoised = c_out * x0 + c_skip * sample

    4) 多步重加噪：ṫ = timesteps[step_index + 1]，若 step_index 不是最后一步
       （prev_timestep 不存在时沿用当前 timestep，该值在最后一步不参与计算）
       ᾱ_prev = alphas_cumprod[ṫ]（ṫ < 0 时取 1.0）
       prev_sample = sqrt(ᾱ_prev) * denoised + sqrt(1 - ᾱ_prev) * noise
       最后一步直接 prev_sample = denoised（不消耗 noise）。

    返回 denoised / prev_sample 为与 sample 同形状的 float32，c_skip / c_out 为 python float。
    """
    sample64 = _tensor("sample", sample)
    output64 = _tensor("model_output", model_output)
    if sample64.shape != output64.shape:
        raise ValueError("sample 与 model_output 形状必须一致")

    if not torch.is_tensor(alphas_cumprod) or not alphas_cumprod.is_floating_point():
        raise ValueError("alphas_cumprod 必须是一维浮点张量")
    if alphas_cumprod.dim() != 1 or alphas_cumprod.numel() == 0:
        raise ValueError("alphas_cumprod 必须是一维且非空")
    alphas = alphas_cumprod.to(torch.float64)
    if not bool(torch.isfinite(alphas).all()):
        raise ValueError("alphas_cumprod 必须全部有限")
    if bool((alphas <= 0).any()) or bool((alphas >= 1).any()):
        raise ValueError("alphas_cumprod 的元素必须落在 (0, 1)")

    if not torch.is_tensor(timesteps) or timesteps.dim() != 1 or timesteps.numel() == 0:
        raise ValueError("timesteps 必须是一维非空张量")
    if timesteps.is_floating_point() or timesteps.is_complex():
        raise ValueError("timesteps 必须是整数张量")
    schedule = [int(v) for v in timesteps.tolist()]
    train_steps = alphas.numel()
    for value in schedule:
        if value < 0 or value >= train_steps:
            raise ValueError(f"timesteps 的取值必须落在 [0, {train_steps - 1}]")

    if isinstance(step_index, bool) or not isinstance(step_index, int):
        raise ValueError("step_index 必须是整数")
    if step_index < 0 or step_index >= len(schedule):
        raise ValueError(f"step_index 必须落在 [0, {len(schedule) - 1}]")

    if prediction_type not in _PREDICTION_TYPES:
        raise ValueError(f"未知的 prediction_type: {prediction_type!r}")
    if not isinstance(clip_sample, bool):
        raise ValueError("clip_sample 必须是 bool")
    if isinstance(clip_sample_range, bool) or not isinstance(clip_sample_range, (int, float)):
        raise ValueError("clip_sample_range 必须是实数")
    clip_range = float(clip_sample_range)
    if not math.isfinite(clip_range) or clip_range <= 0.0:
        raise ValueError("clip_sample_range 必须是有限正数")
    if isinstance(timestep_scaling, bool) or not isinstance(timestep_scaling, (int, float)):
        raise ValueError("timestep_scaling 必须是实数")
    scaling = float(timestep_scaling)
    if not math.isfinite(scaling) or scaling <= 0.0:
        raise ValueError("timestep_scaling 必须是有限正数")

    timestep = schedule[step_index]
    last_index = len(schedule) - 1
    prev_timestep = schedule[step_index + 1] if step_index + 1 < len(schedule) else timestep

    alpha_prod_t = float(alphas[timestep])
    alpha_prod_prev = float(alphas[prev_timestep]) if prev_timestep >= 0 else 1.0

    scaled_timestep = timestep * scaling
    c_skip = _SIGMA_DATA**2 / (scaled_timestep**2 + _SIGMA_DATA**2)
    c_out = scaled_timestep / math.sqrt(scaled_timestep**2 + _SIGMA_DATA**2)

    sqrt_alpha = math.sqrt(alpha_prod_t)
    sqrt_beta = math.sqrt(1.0 - alpha_prod_t)

    if prediction_type == "epsilon":
        x0 = (sample64 - sqrt_beta * output64) / sqrt_alpha
    elif prediction_type == "sample":
        x0 = output64.clone()
    else:
        x0 = sqrt_alpha * sample64 - sqrt_beta * output64

    if clip_sample:
        x0 = x0.clamp(-clip_range, clip_range)

    denoised = c_out * x0 + c_skip * sample64

    if step_index != last_index:
        if noise is None:
            raise ValueError("非最后一步必须显式给出 noise")
        noise64 = _tensor("noise", noise)
        if noise64.shape != sample64.shape:
            raise ValueError("noise 必须与 sample 同形状")
        sqrt_alpha_prev = math.sqrt(alpha_prod_prev)
        sqrt_beta_prev = math.sqrt(max(1.0 - alpha_prod_prev, 0.0))
        prev_sample = sqrt_alpha_prev * denoised + sqrt_beta_prev * noise64
    else:
        prev_sample = denoised

    return denoised.to(torch.float32), prev_sample.to(torch.float32), c_skip, c_out
