import math

import torch


def _scalar(name, value, low=None, high=None, integer=False):
    if integer:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} 必须是整数")
        return int(value)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是实数")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} 必须是有限实数")
    if low is not None and value < low:
        raise ValueError(f"{name} 不能小于 {low}")
    if high is not None and value > high:
        raise ValueError(f"{name} 不能大于 {high}")
    return value


def _bool(name, value):
    if not isinstance(value, bool):
        raise ValueError(f"{name} 必须是 bool")
    return value


def _tensor(name, value, shape=None, allow_none=False):
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{name} 不能为 None")
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} 必须是 torch.Tensor")
    if value.dim() < 2:
        raise ValueError(f"{name} 至少要有两维（第 0 维是 batch）")
    if value.shape[1:].numel() < 2:
        raise ValueError(f"{name} 除 batch 维之外至少要有 2 个元素")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} 必须全为有限值")
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} 的形状必须与 pred_cond 一致")
    return value


def solve(
    pred_cond,
    pred_uncond,
    step,
    num_inference_steps,
    guidance_scale=1.0,
    start=0.0,
    stop=1.0,
    use_original_formulation=False,
    guidance_rescale=0.0,
):
    """AutoGuidance：锚点与失效阈值随表述方式一起翻转的引导算子。"""
    cond = _tensor("pred_cond", pred_cond)
    shape = tuple(cond.shape)

    current = _scalar("step", step, integer=True)
    steps = _scalar("num_inference_steps", num_inference_steps, integer=True)
    if steps <= 0:
        raise ValueError("num_inference_steps 必须为正整数")
    if not 0 <= current < steps:
        raise ValueError("step 必须落在 [0, num_inference_steps) 内")

    scale = _scalar("guidance_scale", guidance_scale)
    start = _scalar("start", start, low=0.0, high=1.0)
    stop = _scalar("stop", stop, low=0.0, high=1.0)
    if start >= 1.0:
        raise ValueError("start 必须严格小于 1")
    if start > stop:
        raise ValueError("stop 不能小于 start")
    original = _bool("use_original_formulation", use_original_formulation)
    rescale = _scalar("guidance_rescale", guidance_rescale, low=0.0, high=1.0)

    within = int(start * steps) <= current < int(stop * steps)
    identity = 0.0 if original else 1.0
    ag_on = within and not math.isclose(scale, identity)

    c = cond.to(torch.float64)
    if ag_on:
        uncond = _tensor("pred_uncond", pred_uncond, shape, allow_none=True)
        if uncond is None:
            raise ValueError("引导分量生效时必须提供 pred_uncond")
        u = uncond.to(torch.float64)
        shift = c - u
        pred = (c if original else u) + scale * shift
    else:
        pred = c

    if rescale > 0.0:
        dims = list(range(1, c.dim()))
        std_text = c.std(dim=dims, keepdim=True)
        std_cfg = pred.std(dim=dims, keepdim=True)
        pred = rescale * (pred * (std_text / std_cfg)) + (1.0 - rescale) * pred

    return pred.to(torch.float32)
