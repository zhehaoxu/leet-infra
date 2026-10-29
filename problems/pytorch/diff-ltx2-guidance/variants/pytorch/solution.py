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
    pred_stg,
    pred_modality,
    step,
    num_inference_steps,
    cfg_scale=1.0,
    stg_scale=0.0,
    modality_scale=1.0,
    start=0.0,
    stop=1.0,
):
    """LTX-2 三路增量引导的单模态合成。

    四段同形预测（第 0 维为 batch）：
      pred_cond       —— 条件预测（基准）；
      pred_uncond     —— 无条件预测，配合 cfg_scale；
      pred_stg        —— 扰动若干 transformer block 后的条件预测，配合 stg_scale；
      pred_modality   —— 关掉 A2V/V2A 跨模态注意力后的条件预测，配合 modality_scale。

    三个尺度共用同一个左闭右开窗口：
      within = int(start*N) <= step < int(stop*N)，N = num_inference_steps
    各分量是否生效：
      cfg_on = within and not isclose(cfg_scale, 1.0)
      stg_on = within and not isclose(stg_scale, 0.0)
      mod_on = within and not isclose(modality_scale, 1.0)

    合成采用增量（delta）形式，三项都以 pred_cond 为基准、彼此可加：
      pred = pred_cond
             + (cfg_scale - 1)      * (pred_cond - pred_uncond)      # cfg_on 时
             +  stg_scale           * (pred_cond - pred_stg)         # stg_on 时
             + (modality_scale - 1) * (pred_cond - pred_modality)    # mod_on 时
    注意 cfg 与 modality 是「以 1 为恒等」的尺度（差的是 1），stg 是「以 0 为恒等」的尺度。

    某个分量关闭时对应输入可以传 None；若分量生效却传 None，抛 ValueError。
    返回 float32 张量，形状与 pred_cond 相同。
    """
    cond = _tensor("pred_cond", pred_cond)
    if cond is None:
        raise ValueError("pred_cond 不能为 None")
    shape = tuple(cond.shape)
    uncond = _tensor("pred_uncond", pred_uncond, shape)
    stg = _tensor("pred_stg", pred_stg, shape)
    modality = _tensor("pred_modality", pred_modality, shape)

    cfg = _scalar("cfg_scale", cfg_scale)
    stg_scale = _scalar("stg_scale", stg_scale)
    modality_scale = _scalar("modality_scale", modality_scale)
    start = _scalar("start", start)
    stop = _scalar("stop", stop)
    if not 0.0 <= start <= 1.0 or not 0.0 <= stop <= 1.0:
        raise ValueError("start / stop 必须落在 [0, 1] 内")
    if start > stop:
        raise ValueError("start 不能大于 stop")

    current = _integer("step", step)
    steps = _integer("num_inference_steps", num_inference_steps)
    if steps <= 0:
        raise ValueError("num_inference_steps 必须为正整数")
    if not 0 <= current < steps:
        raise ValueError("step 必须落在 [0, num_inference_steps) 内")

    within = int(start * steps) <= current < int(stop * steps)
    cfg_on = within and not math.isclose(cfg, 1.0)
    stg_on = within and not math.isclose(stg_scale, 0.0)
    mod_on = within and not math.isclose(modality_scale, 1.0)

    if cfg_on and uncond is None:
        raise ValueError("CFG 分量生效时必须提供 pred_uncond")
    if stg_on and stg is None:
        raise ValueError("STG 分量生效时必须提供 pred_stg")
    if mod_on and modality is None:
        raise ValueError("模态隔离分量生效时必须提供 pred_modality")

    c = cond.to(torch.float64)
    pred = c
    if cfg_on:
        pred = pred + (cfg - 1.0) * (c - uncond.to(torch.float64))
    if stg_on:
        pred = pred + stg_scale * (c - stg.to(torch.float64))
    if mod_on:
        pred = pred + (modality_scale - 1.0) * (c - modality.to(torch.float64))

    return pred.to(torch.float32)
