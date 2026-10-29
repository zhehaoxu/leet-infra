import math

import torch

_PREDICTION_TYPES = ("epsilon", "v_prediction")


def _float_tensor(name, value):
    if not torch.is_tensor(value):
        raise ValueError(f"{name} 必须是 torch 张量")
    if not value.is_floating_point():
        raise ValueError(f"{name} 必须是浮点张量")
    if value.numel() == 0:
        raise ValueError(f"{name} 不能为空张量")
    return value.to(torch.float64)


def solve(sample, model_output, sigmas, step_index, prediction_type="epsilon", noise=None):
    """EulerAncestral 单步：返回 (scaled_input, prev_sample, pred_original_sample, sigma_up, sigma_down)。

    sample / model_output:同形状浮点张量；sigmas:一维浮点张量，长度 K+1、有限非负且单调不增；
    step_index:整数，落在 [0, K-1]；当前步的 sigma 必须为正。

    1) 模型输入端：scaled_input = sample / sqrt(sigma**2 + 1)（上游 scale_model_input）。

    2) 预测 x0（注意 v_prediction 支照抄上游实现）：
       epsilon      -> x0 = sample - sigma * model_output
       v_prediction -> x0 = model_output * (-sigma / sqrt(sigma**2 + 1)) + sample / (sigma**2 + 1)

    3) 祖先采样把 sigma 的下降拆成确定项与随机项：
       sigma_from = sigmas[step_index]，sigma_to = sigmas[step_index + 1]
       sigma_up   = sqrt(sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2)
       sigma_down = sqrt(sigma_to**2 - sigma_up**2)

    4) Euler 步 + 注入噪声：
       derivative = (sample - x0) / sigma_from
       prev_sample = sample + derivative * (sigma_down - sigma_from) + noise * sigma_up

    noise 是显式入参（与 sample 同形状的浮点张量）；sigma_up = 0 时完全不参与计算，
    此时可以不传。返回前三个 float32 张量与 sample 同形状，sigma_up / sigma_down 是 python float。
    """
    sample64 = _float_tensor("sample", sample)
    output64 = _float_tensor("model_output", model_output)
    if sample64.shape != output64.shape:
        raise ValueError("sample 与 model_output 形状必须一致")
    if not torch.is_tensor(sigmas) or not sigmas.is_floating_point() or sigmas.dim() != 1:
        raise ValueError("sigmas 必须是一维浮点张量")
    if sigmas.numel() < 2:
        raise ValueError("sigmas 长度至少为 2")
    values = sigmas.to(torch.float64)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("sigmas 必须全部有限")
    if bool((values < 0).any()):
        raise ValueError("sigmas 必须非负")
    if bool((values[1:] > values[:-1]).any()):
        raise ValueError("sigmas 必须单调不增")

    last = values.numel() - 1
    if isinstance(step_index, bool) or not isinstance(step_index, int):
        raise ValueError("step_index 必须是整数")
    if step_index < 0 or step_index > last - 1:
        raise ValueError(f"step_index 必须落在 [0, {last - 1}]")

    if prediction_type not in _PREDICTION_TYPES:
        raise ValueError(f"未知的 prediction_type: {prediction_type!r}")

    sigma_from = float(values[step_index])
    sigma_to = float(values[step_index + 1])
    if sigma_from <= 0.0:
        raise ValueError("当前步的 sigma 必须为正")

    sigma_up = math.sqrt(sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2)
    sigma_down = math.sqrt(max(sigma_to**2 - sigma_up**2, 0.0))

    scaled_input = sample64 / math.sqrt(sigma_from**2 + 1.0)

    if prediction_type == "epsilon":
        pred_original = sample64 - sigma_from * output64
    else:
        pred_original = output64 * (-sigma_from / math.sqrt(sigma_from**2 + 1.0)) + sample64 / (
            sigma_from**2 + 1.0
        )

    derivative = (sample64 - pred_original) / sigma_from
    prev = sample64 + derivative * (sigma_down - sigma_from)

    if sigma_up > 0.0:
        if noise is None:
            raise ValueError("sigma_up > 0 时必须显式给出 noise")
        noise64 = _float_tensor("noise", noise)
        if noise64.shape != sample64.shape:
            raise ValueError("noise 必须与 sample 同形状")
        prev = prev + noise64 * sigma_up

    return (
        scaled_input.to(torch.float32),
        prev.to(torch.float32),
        pred_original.to(torch.float32),
        sigma_up,
        sigma_down,
    )
