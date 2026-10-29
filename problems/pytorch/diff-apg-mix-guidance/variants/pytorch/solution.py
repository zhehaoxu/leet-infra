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


def _integer(name, value, low=None):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} 必须是整数")
    value = int(value)
    if low is not None and value < low:
        raise ValueError(f"{name} 不能小于 {low}")
    return value


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


def _normalize_rows(tensor, dims):
    norm = tensor.norm(p=2, dim=dims, keepdim=True).clamp_min(1e-12)
    return tensor / norm


def solve(
    pred_cond,
    pred_uncond,
    step,
    num_inference_steps,
    buffer=None,
    cfg_scale=3.5,
    apg_scale=10.0,
    momentum=-0.5,
    norm_threshold=10.0,
    eta=0.0,
    apg_start_step=5,
    start=0.0,
    stop=1.0,
    use_original_formulation=False,
):
    """APG-Mix：先跑 CFG 预热动量，再切到投影引导。

    两段同形预测（第 0 维为 batch）：pred_cond 条件预测、pred_uncond 无条件预测。
    buffer 是跨步携带的动量状态（同形张量），首次调用传 None 表示从 0 起步；
    momentum 为 None 时不做动量，此时 buffer 必须是 None。

    先判两件事（N = num_inference_steps，s_cfg = cfg_scale，s_apg = apg_scale）：
      cfg_on = [int(start*N) <= step < int(stop*N)] and not isclose(s_cfg, 1.0)
      apg_on = cfg_on and step > apg_start_step and not isclose(s_apg, 1.0)

    三个分支：
      not cfg_on -> pred = pred_cond，buffer 原样带回（不更新）
      cfg_on 且 not apg_on -> 先用 d = pred_cond - pred_uncond 更新动量（若有），
                              再 pred = pred_uncond + s_cfg * d
      apg_on -> 用 d 更新动量后取 d = 动量值（无动量时 d 就是 pred_cond - pred_uncond），
                对 d 做范数截断（norm_threshold > 0 时乘 min(1, norm_threshold/‖d‖₂)，
                范数在除 batch 外的所有维度上求），
                再把 d 分解为沿 pred_cond 单位方向的分量与正交分量：
                  u = pred_cond / ‖pred_cond‖
                  d_parallel = <d, u> * u， d_orth = d - d_parallel
                取 normalized = d_orth + eta * d_parallel，
                pred = pred_uncond + s_apg * normalized

    动量更新公式：running = d + momentum * running_prev（running_prev 为 0 表示还未初始化）。

    use_original_formulation=True 时基准预测从 pred_uncond 换成 pred_cond，且「尺度退化」
    的判据也从 isclose(s, 1.0) 换成 isclose(s, 0.0)（CFG 与 APG 两个尺度都如此）。
    返回 (pred, new_buffer)：均为 float32，new_buffer 与 pred_cond 同形；
    无动量时 new_buffer 为 None。输入 buffer 不被就地修改。
    """
    cond = _tensor("pred_cond", pred_cond)
    if cond is None:
        raise ValueError("pred_cond 不能为 None")
    shape = tuple(cond.shape)
    uncond = _tensor("pred_uncond", pred_uncond, shape)
    state = _tensor("buffer", buffer, shape)

    cfg = _scalar("cfg_scale", cfg_scale, 0.0)
    apg = _scalar("apg_scale", apg_scale, 0.0)
    threshold = _scalar("norm_threshold", norm_threshold, 0.0)
    eta = _scalar("eta", eta, 0.0)
    start = _scalar("start", start)
    stop = _scalar("stop", stop)
    if not 0.0 <= start <= 1.0 or not 0.0 <= stop <= 1.0:
        raise ValueError("start / stop 必须落在 [0, 1] 内")
    if start > stop:
        raise ValueError("start 不能大于 stop")

    if momentum is None:
        if state is not None:
            raise ValueError("momentum 为 None 时不接受 buffer")
        mu = None
    else:
        mu = _scalar("momentum", momentum)

    current = _integer("step", step, 0)
    steps = _integer("num_inference_steps", num_inference_steps, 1)
    if current >= steps:
        raise ValueError("step 必须落在 [0, num_inference_steps) 内")
    apg_start = _integer("apg_start_step", apg_start_step, 0)

    off = 0.0 if use_original_formulation else 1.0
    cfg_on = int(start * steps) <= current < int(stop * steps) and not math.isclose(cfg, off)
    apg_on = cfg_on and current > apg_start and not math.isclose(apg, off)

    if cfg_on and uncond is None:
        raise ValueError("CFG 生效时必须提供 pred_uncond")

    c = cond.to(torch.float64)

    if not cfg_on:
        return c.to(torch.float32), None if state is None else state.to(torch.float32)

    kept_state = None
    if mu is not None:
        kept_state = torch.zeros_like(c) if state is None else state.to(torch.float64)

    u = uncond.to(torch.float64)
    diff = c - u
    if mu is not None:
        running = diff + mu * kept_state
    else:
        running = diff

    base = c if use_original_formulation else u
    if not apg_on:
        pred = base + cfg * diff
    else:
        dims = tuple(range(1, c.dim()))
        update = running
        if threshold > 0.0:
            norm = update.norm(p=2, dim=dims, keepdim=True).clamp_min(1e-12)
            update = update * torch.clamp(threshold / norm, max=1.0)
        direction = _normalize_rows(c, dims)
        parallel = (update * direction).sum(dim=dims, keepdim=True) * direction
        normalized = (update - parallel) + eta * parallel
        pred = base + apg * normalized

    new_buffer = None if mu is None else running.to(torch.float32)
    return pred.to(torch.float32), new_buffer
