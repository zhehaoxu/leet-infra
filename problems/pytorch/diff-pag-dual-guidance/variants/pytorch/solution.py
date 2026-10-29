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


def _integer(name, value):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} 必须是整数")
    return int(value)


def _window(name, start, stop, num_inference_steps):
    lo = _scalar(f"{name}_start", start)
    hi = _scalar(f"{name}_stop", stop)
    if not 0.0 <= lo <= 1.0 or not 0.0 <= hi <= 1.0:
        raise ValueError(f"{name} 的 start / stop 必须落在 [0, 1] 内")
    if lo > hi:
        raise ValueError(f"{name} 的 start 不能大于 stop")
    return int(lo * num_inference_steps), int(hi * num_inference_steps)


def _tensor(name, value, shape=None):
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} 必须是 torch.Tensor")
    if value.dim() < 1:
        raise ValueError(f"{name} 至少要有一维 batch")
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} 的形状必须与 pred_cond 一致")
    return value


def solve(
    pred_cond,
    pred_uncond,
    pred_cond_skip,
    guidance_scale,
    perturbed_scale,
    step,
    num_inference_steps,
    start=0.0,
    stop=1.0,
    perturbed_start=0.01,
    perturbed_stop=0.2,
):
    """PAG 双尺度引导的四种分支。

    三段预测同形状（第 0 维为 batch）：
      pred_cond        —— 条件预测；
      pred_uncond      —— 无条件预测，配合 guidance_scale 做 CFG；
      pred_cond_skip   —— 扰动注意力分支的条件预测，配合 perturbed_scale 做 PAG。

    step / num_inference_steps 决定两套窗口：
      CFG 窗口为左闭右开的整数区间 [int(start*N), int(stop*N))；
      PAG 窗口为开区间 (int(perturbed_start*N), int(perturbed_stop*N))。
    另外 guidance_scale 与 1.0 相近、perturbed_scale 与 0.0 相近时，对应分支视为关闭。

    四个分支（与 diffusers 一致）：
      两者关闭            -> pred_cond
      仅 PAG 开启         -> pred_cond_skip + perturbed_scale * (pred_cond - pred_cond_skip)
      仅 CFG 开启         -> pred_uncond + guidance_scale * (pred_cond - pred_uncond)
      两者开启            -> pred_uncond + guidance_scale * (pred_cond - pred_uncond)
                                        + perturbed_scale * (pred_cond - pred_cond_skip)

    某个分支关闭时对应的输入可以为 None；但若该分支生效却传 None，抛 ValueError。
    返回 float32 张量，形状与 pred_cond 相同。
    """
    cond = _tensor("pred_cond", pred_cond)
    if cond is None:
        raise ValueError("pred_cond 不能为 None")
    shape = tuple(cond.shape)
    uncond = _tensor("pred_uncond", pred_uncond, shape)
    skip = _tensor("pred_cond_skip", pred_cond_skip, shape)

    scale = _scalar("guidance_scale", guidance_scale, 0.0)
    skip_scale = _scalar("perturbed_scale", perturbed_scale, 0.0)

    current = _integer("step", step)
    steps = _integer("num_inference_steps", num_inference_steps)
    if steps <= 0:
        raise ValueError("num_inference_steps 必须为正整数")
    if not 0 <= current < steps:
        raise ValueError("step 必须落在 [0, num_inference_steps) 内")

    cfg_lo, cfg_hi = _window("start/stop", start, stop, steps)
    pag_lo, pag_hi = _window("perturbed", perturbed_start, perturbed_stop, steps)

    cfg_on = cfg_lo <= current < cfg_hi and not math.isclose(scale, 1.0)
    pag_on = pag_lo < current < pag_hi and not math.isclose(skip_scale, 0.0)

    if cfg_on and uncond is None:
        raise ValueError("CFG 分支生效时必须提供 pred_uncond")
    if pag_on and skip is None:
        raise ValueError("PAG 分支生效时必须提供 pred_cond_skip")

    c = cond.to(torch.float64)
    if not cfg_on and not pag_on:
        pred = c
    elif not cfg_on:
        k = skip.to(torch.float64)
        pred = k + skip_scale * (c - k)
    elif not pag_on:
        u = uncond.to(torch.float64)
        pred = u + scale * (c - u)
    else:
        u = uncond.to(torch.float64)
        k = skip.to(torch.float64)
        pred = u + scale * (c - u) + skip_scale * (c - k)

    return pred.to(torch.float32)
