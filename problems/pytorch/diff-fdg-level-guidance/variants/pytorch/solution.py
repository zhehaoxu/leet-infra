import math

import torch

_NORM_EPS = 1e-12


def _sequence(name, value, length=None):
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} 必须是 list 或 tuple")
    if length is not None and len(value) != length:
        raise ValueError(f"{name} 的长度必须与频率层数一致")
    return list(value)


def _scalar(name, value, low=None, high=None):
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


def _level_pair(name, values):
    levels = _sequence(name, values)
    if not levels:
        raise ValueError(f"{name} 不能为空")
    shape = None
    for level in levels:
        if not isinstance(level, torch.Tensor):
            raise ValueError(f"{name} 的每个频率层都必须是 torch.Tensor")
        if level.dim() < 2:
            raise ValueError(f"{name} 的每个频率层至少要两维：第 0 维 batch，其余维度参与投影与统计")
        if shape is None:
            shape = tuple(level.shape)
        elif tuple(level.shape) != shape:
            raise ValueError(f"{name} 的所有频率层必须同形状")
    return levels


def _rescale(pred, reference, strength):
    red = tuple(range(1, pred.dim()))
    std_ref = reference.std(dim=red, keepdim=True)
    std_pred = pred.std(dim=red, keepdim=True)
    if (std_pred == 0).any() or not torch.isfinite(std_pred).all():
        raise ValueError("开启 rescale 时该频率层每个样本的预测不能是常数")
    scaled = pred * (std_ref / std_pred)
    return strength * scaled + (1.0 - strength) * pred


def solve(
    cond_levels,
    uncond_levels,
    guidance_scales,
    parallel_weights=None,
    level_rescale=None,
    active_levels=None,
):
    """FDG：逐频层的引导、可选的平行分量加权与逐层 rescale。

    cond_levels / uncond_levels:长度相同的张量列表，按「高频 → 低频」排列，
        每个频率层形状为 (batch, ...)，所有层同形状；
    guidance_scales:每层的引导尺度 s_l（非负有限实数）；
    parallel_weights:可选的每层平行分量权重 w_l（非负有限实数，缺省全为 1.0）；
    level_rescale:可选的每层 rescale 强度 φ_l ∈ [0, 1]（缺省全为 0.0）；
    active_levels:可选的每层开关（bool 列表，缺省全为 True）。

    对每个频率层 l：
      未开启            -> pred_l = cond_l
      否则              -> shift = cond_l - uncond_l
                          若 w_l 与 1.0 不相近：
                              p = (shift·ĉ)ĉ，o = shift - p，shift = w_l·p + o
                          pred_l = uncond_l + s_l · shift
                          若 φ_l > 0：按标准差的比值把 pred_l 拉回 cond_l 的尺度，再与未缩放
                          结果按 (1-φ_l, φ_l) 混合（标准差为 torch 默认的无偏估计）
    最终返回所有 pred_l 之和（对应反变换的求和步骤），float32。
    """
    cond = _level_pair("cond_levels", cond_levels)
    uncond = _level_pair("uncond_levels", uncond_levels)
    if len(cond) != len(uncond):
        raise ValueError("cond_levels 与 uncond_levels 的频率层数必须一致")
    for left, right in zip(cond, uncond):
        if tuple(left.shape) != tuple(right.shape):
            raise ValueError("同一频率层的 cond / uncond 必须同形状")

    num_levels = len(cond)
    scales = _sequence("guidance_scales", guidance_scales, num_levels)
    scales = [_scalar(f"guidance_scales[{i}]", v, low=0.0) for i, v in enumerate(scales)]

    if parallel_weights is None:
        weights = [1.0] * num_levels
    else:
        raw = _sequence("parallel_weights", parallel_weights, num_levels)
        weights = [_scalar(f"parallel_weights[{i}]", v, low=0.0) for i, v in enumerate(raw)]

    if level_rescale is None:
        rescales = [0.0] * num_levels
    else:
        raw = _sequence("level_rescale", level_rescale, num_levels)
        rescales = [_scalar(f"level_rescale[{i}]", v, low=0.0, high=1.0) for i, v in enumerate(raw)]

    if active_levels is None:
        active = [True] * num_levels
    else:
        raw = _sequence("active_levels", active_levels, num_levels)
        for i, v in enumerate(raw):
            if not isinstance(v, bool):
                raise ValueError(f"active_levels[{i}] 必须是 bool")
        active = list(raw)

    red = tuple(range(1, cond[0].dim()))
    total = None
    for index in range(num_levels):
        c = cond[index].to(torch.float64)
        u = uncond[index].to(torch.float64)

        if not active[index]:
            pred = c
        else:
            shift = c - u
            weight = weights[index]
            if not math.isclose(weight, 1.0):
                unit = c / torch.clamp(
                    torch.linalg.vector_norm(c, ord=2, dim=red, keepdim=True), min=_NORM_EPS
                )
                parallel = torch.sum(shift * unit, dim=red, keepdim=True) * unit
                shift = weight * parallel + (shift - parallel)
            pred = u + scales[index] * shift
            if rescales[index] > 0.0:
                pred = _rescale(pred, c, rescales[index])

        total = pred if total is None else total + pred

    return total.to(torch.float32)
