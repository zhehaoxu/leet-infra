import math

import torch

_DATA_ALGORITHMS = ("dpmsolver++", "sde-dpmsolver++")
_NOISE_ALGORITHMS = ("dpmsolver", "sde-dpmsolver")
_ALGORITHMS = _DATA_ALGORITHMS + _NOISE_ALGORITHMS
_VARIANCE_TYPES = ("fixed_small", "fixed_large", "learned", "learned_range")
_PREDICTION_TYPES = ("epsilon", "sample", "v_prediction", "flow_prediction")


def _check_tensor(name, value):
    if not torch.is_tensor(value):
        raise ValueError(f"{name} 必须是 torch 张量")
    if not value.is_floating_point():
        raise ValueError(f"{name} 必须是浮点张量")
    if value.dim() < 2:
        raise ValueError(f"{name} 至少要有 2 维（batch, channels, ...）")
    if value.numel() == 0:
        raise ValueError(f"{name} 不能为空张量")
    return value.to(torch.float64)


def _quantile_linear(sorted_values, q):
    count = sorted_values.shape[1]
    position = q * (count - 1)
    low = int(math.floor(position))
    high = min(low + 1, count - 1)
    frac = position - low
    return sorted_values[:, low] * (1.0 - frac) + sorted_values[:, high] * frac


def _threshold(sample, ratio, sample_max_value):
    batch = sample.shape[0]
    flat = sample.reshape(batch, -1).abs()
    flat, _ = flat.sort(dim=1)
    scale = _quantile_linear(flat, ratio).clamp(min=1.0, max=sample_max_value).unsqueeze(1)
    clipped = sample.reshape(batch, -1).clamp(-scale, scale) / scale
    return clipped.reshape_as(sample)


def solve(
    model_output,
    sample,
    sigma,
    prediction_type="epsilon",
    algorithm_type="dpmsolver++",
    variance_type="fixed_small",
    thresholding=False,
    dynamic_thresholding_ratio=0.995,
    sample_max_value=1.0,
    use_flow_sigmas=False,
):
    """把网络裸输出换成 DPM-Solver 需要的口径，并按需做动态阈值：返回同形状 float32 张量。

    model_output / sample:浮点张量，至少 2 维；sample 的形状是 (B, C, ...)。
    sigma:当前噪声级，有限正数（use_flow_sigmas=True 时要求 0 < sigma < 1）。

    1) α/σ_t 换算：
       use_flow_sigmas=True  -> α = 1 - σ，σ_t = σ
       否则                  -> α = 1/sqrt(σ² + 1)，σ_t = σ·α

    2) algorithm_type 决定输出是 x0 还是 ε：
       "dpmsolver++" / "sde-dpmsolver++"（数据口径）：
         epsilon        -> x0 = (sample - σ_t·ε) / α
         sample         -> x0 = model_output
         v_prediction   -> x0 = α·sample - σ_t·model_output
         flow_prediction-> x0 = sample - σ·model_output（用原始 σ，不走 α/σ_t）
         再按需 x0 = 动态阈值(x0)
       "dpmsolver" / "sde-dpmsolver"（噪声口径）：
         epsilon        -> ε = model_output
         sample         -> ε = (sample - α·model_output) / σ_t
         v_prediction   -> ε = α·model_output + σ_t·sample
         flow_prediction-> 不支持，抛 ValueError
         再按需：x0 = (sample - σ_t·ε)/α → 阈值 → ε = (sample - α·x0)/σ_t

    3) 方差通道：variance_type 为 "learned"/"learned_range" 时只对 epsilon 预测生效，
       此时 model_output 是 (B, 6, ...) 而 sample 是 (B, 3, ...)，只取前 3 个通道当 ε。

    4) 动态阈值（thresholding=True）：
       对每个样本把 sample 展平成 (B, -1) 并取 |x| 的 dynamic_thresholding_ratio 分位数 s
       （线性插值），钳制到 [1, sample_max_value]，再把 x 钳到 [-s, s] 后除以 s。

    返回与 sample 同形状的 float32 张量。
    """
    output64 = _check_tensor("model_output", model_output)
    sample64 = _check_tensor("sample", sample)

    if isinstance(sigma, bool) or not isinstance(sigma, (int, float)):
        raise ValueError("sigma 必须是实数")
    sigma = float(sigma)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma 必须是有限正数")
    if prediction_type not in _PREDICTION_TYPES:
        raise ValueError(f"未知的 prediction_type: {prediction_type!r}")
    if algorithm_type not in _ALGORITHMS:
        raise ValueError(f"未知的 algorithm_type: {algorithm_type!r}")
    if variance_type not in _VARIANCE_TYPES:
        raise ValueError(f"未知的 variance_type: {variance_type!r}")
    if not isinstance(thresholding, bool):
        raise ValueError("thresholding 必须是 bool")
    if not isinstance(use_flow_sigmas, bool):
        raise ValueError("use_flow_sigmas 必须是 bool")
    if isinstance(dynamic_thresholding_ratio, bool) or not isinstance(
        dynamic_thresholding_ratio, (int, float)
    ):
        raise ValueError("dynamic_thresholding_ratio 必须是实数")
    ratio = float(dynamic_thresholding_ratio)
    if not math.isfinite(ratio) or ratio <= 0.0 or ratio > 1.0:
        raise ValueError("dynamic_thresholding_ratio 必须落在 (0, 1]")
    if isinstance(sample_max_value, bool) or not isinstance(sample_max_value, (int, float)):
        raise ValueError("sample_max_value 必须是实数")
    max_value = float(sample_max_value)
    if not math.isfinite(max_value) or max_value < 1.0:
        raise ValueError("sample_max_value 必须 >= 1（与 min=1 的钳位配合）")

    if use_flow_sigmas and sigma >= 1.0:
        raise ValueError("use_flow_sigmas=True 时 sigma 必须落在 (0, 1)")

    learned = variance_type in ("learned", "learned_range")
    if learned:
        if prediction_type != "epsilon":
            raise ValueError("learned / learned_range 方差只支持 epsilon 预测")
        if output64.dim() != sample64.dim() or output64.shape[1] != 6 or sample64.shape[1] != 3:
            raise ValueError("learned 方差要求 model_output 为 (B, 6, ...) 且 sample 为 (B, 3, ...)")
        if output64.shape[0] != sample64.shape[0] or output64.shape[2:] != sample64.shape[2:]:
            raise ValueError("model_output 与 sample 的 batch / 空间维必须一致")
        core = output64[:, :3]
    else:
        if output64.shape != sample64.shape:
            raise ValueError("model_output 与 sample 形状必须一致")
        core = output64

    if use_flow_sigmas:
        alpha = 1.0 - sigma
        sigma_t = sigma
    else:
        alpha = 1.0 / math.sqrt(sigma**2 + 1.0)
        sigma_t = sigma * alpha

    if algorithm_type in _DATA_ALGORITHMS:
        if prediction_type == "epsilon":
            out = (sample64 - sigma_t * core) / alpha
        elif prediction_type == "sample":
            out = core
        elif prediction_type == "v_prediction":
            out = alpha * sample64 - sigma_t * core
        else:
            out = sample64 - sigma * core
        if thresholding:
            out = _threshold(out, ratio, max_value)
    else:
        if prediction_type == "epsilon":
            eps = core
        elif prediction_type == "sample":
            eps = (sample64 - alpha * core) / sigma_t
        elif prediction_type == "v_prediction":
            eps = alpha * core + sigma_t * sample64
        else:
            raise ValueError("噪声口径（dpmsolver / sde-dpmsolver）不支持 flow_prediction")
        if thresholding:
            x0 = (sample64 - sigma_t * eps) / alpha
            x0 = _threshold(x0, ratio, max_value)
            eps = (sample64 - alpha * x0) / sigma_t
        out = eps

    return out.to(torch.float32)
