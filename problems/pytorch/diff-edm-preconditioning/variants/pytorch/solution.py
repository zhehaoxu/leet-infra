import math

import torch

_PREDICTION_TYPES = ("epsilon", "v_prediction")
SQRT2_MINUS_1 = math.sqrt(2.0) - 1.0


def _int(name, value, low, high):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} 必须是整数")
    if value < low or value > high:
        raise ValueError(f"{name} 必须落在 [{low}, {high}]")
    return value


def _non_negative(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是数字")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} 必须是有限非负数")
    return value


def solve(
    sample,
    model_output,
    sigmas,
    step_index,
    sigma_data=0.5,
    prediction_type="epsilon",
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
    noise=None,
):
    """EDM 预条件 + Euler 单步：返回 (model_input, c_noise, prev_sample, denoised, sigma_hat)。

    sample / model_output:同形状浮点张量，model_output 是网络在预条件输入域的裸输出 F(·)；
    sigmas:一维浮点张量，长度 K+1、非增非负，最后一项通常是 0；
    step_index:整数，落在 [0, K-1]；sigma_data:训练数据标准差 σ_d，有限正数。

    记 σ = sigmas[step_index]，σ̂ 为加噪后的噪声级：
      gamma = min(s_churn / K, sqrt(2) - 1)（仅当 s_tmin <= σ <= s_tmax，否则 0；K = len(sigmas) - 1）
      σ̂ = σ · (1 + gamma)
      gamma > 0 时先加噪：sample ← sample + noise · s_noise · sqrt(σ̂² - σ²)（noise 必须显式给出）

    网络输入侧（用**未加噪**的 σ）：
      c_in    = 1 / sqrt(σ² + σ_d²)
      c_noise = 0.25 · ln σ
      model_input = 输入 sample · c_in

    输出合成侧（用**加噪后**的 σ̂）：
      c_skip = σ_d² / (σ̂² + σ_d²)
      c_out  = ±σ̂·σ_d / sqrt(σ̂² + σ_d²)（epsilon 取 +，v_prediction 取 -）
      denoised = c_skip · sample + c_out · model_output

    Euler 单步：
      derivative = (sample - denoised) / σ̂
      dt = sigmas[step_index + 1] - σ̂
      prev_sample = sample + derivative · dt

    返回 (model_input, c_noise, prev_sample, denoised, sigma_hat)：
    前三个张量与 sample 同形状的 float32，c_noise 与 sigma_hat 为 python float。
    """
    if not torch.is_tensor(sample) or not torch.is_tensor(model_output):
        raise ValueError("sample 与 model_output 必须是 torch 张量")
    if not sample.is_floating_point() or not model_output.is_floating_point():
        raise ValueError("sample 与 model_output 必须是浮点张量")
    if sample.shape != model_output.shape or sample.numel() == 0:
        raise ValueError("sample 与 model_output 必须同形状且非空")
    if not torch.is_tensor(sigmas) or sigmas.dim() != 1 or not sigmas.is_floating_point():
        raise ValueError("sigmas 必须是一维浮点张量")
    if sigmas.numel() < 2:
        raise ValueError("sigmas 至少需要两个元素")
    sigma_values = sigmas.to(torch.float64)
    if not bool(torch.isfinite(sigma_values).all()) or bool((sigma_values < 0).any()):
        raise ValueError("sigmas 必须有限且非负")
    if bool((sigma_values[1:] > sigma_values[:-1]).any()):
        raise ValueError("sigmas 必须单调不增")

    num_steps = sigma_values.numel() - 1
    index = _int("step_index", step_index, 0, num_steps - 1)

    sigma_data = _non_negative("sigma_data", sigma_data)
    if sigma_data == 0.0:
        raise ValueError("sigma_data 必须是正数")
    s_churn = _non_negative("s_churn", s_churn)
    s_noise = _non_negative("s_noise", s_noise)
    low = _non_negative("s_tmin", s_tmin)
    high = s_tmax
    if isinstance(high, bool) or not isinstance(high, (int, float)) or math.isnan(float(high)):
        raise ValueError("s_tmax 必须是数字")
    high = float(high)
    if low > high:
        raise ValueError("s_tmin 不能大于 s_tmax")
    if prediction_type not in _PREDICTION_TYPES:
        raise ValueError(f"未知的 prediction_type: {prediction_type!r}")

    sigma = float(sigma_values[index])
    if sigma <= 0.0:
        raise ValueError("当前步的 sigma 必须为正")

    gamma = min(s_churn / num_steps, SQRT2_MINUS_1) if low <= sigma <= high else 0.0
    sigma_hat = sigma * (1.0 + gamma)

    sample64 = sample.to(torch.float64)
    output64 = model_output.to(torch.float64)

    c_in = 1.0 / math.sqrt(sigma**2 + sigma_data**2)
    c_noise = 0.25 * math.log(sigma)
    model_input = sample64 * c_in

    if gamma > 0.0:
        if noise is None:
            raise ValueError("gamma > 0 时必须显式提供 noise")
        if not torch.is_tensor(noise) or not noise.is_floating_point() or noise.shape != sample.shape:
            raise ValueError("noise 必须是和 sample 同形状的浮点张量")
        scale = math.sqrt(sigma_hat**2 - sigma**2)
        sample64 = sample64 + noise.to(torch.float64) * s_noise * scale

    c_skip = sigma_data**2 / (sigma_hat**2 + sigma_data**2)
    magnitude = sigma_hat * sigma_data / math.sqrt(sigma_hat**2 + sigma_data**2)
    c_out = magnitude if prediction_type == "epsilon" else -magnitude
    denoised = c_skip * sample64 + c_out * output64

    derivative = (sample64 - denoised) / sigma_hat
    dt = float(sigma_values[index + 1]) - sigma_hat
    prev_sample = sample64 + derivative * dt

    return (
        model_input.to(torch.float32),
        float(c_noise),
        prev_sample.to(torch.float32),
        denoised.to(torch.float32),
        float(sigma_hat),
    )
