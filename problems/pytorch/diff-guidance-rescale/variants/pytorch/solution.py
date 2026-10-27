import math

import torch


def _std_dims(dims, ndim):
    if dims is None:
        dims = tuple(range(1, ndim))
    else:
        if isinstance(dims, int):
            dims = (dims,)
        dims = tuple(dims)
    if not dims:
        raise ValueError("dims 不能为空：至少需要一个可求标准差的维度")
    for d in dims:
        if isinstance(d, bool) or not isinstance(d, int):
            raise ValueError("dims 只能包含整数维度")
        if d == 0:
            raise ValueError("dims 不能包含第 0 维（要为每个样本单独算标准差）")
        if d < 0 or d >= ndim:
            raise ValueError(f"维度 {d} 超出范围（ndim={ndim}）")
    if len(set(dims)) != len(dims):
        raise ValueError("dims 里有重复维度")
    return dims


def solve(eps_cond, eps_uncond, guidance_scale, t, t_low=0.0, t_high=1.0, rescale=1.0, dims=None):
    if not torch.is_tensor(eps_cond) or not torch.is_tensor(eps_uncond):
        raise ValueError("eps_cond 与 eps_uncond 必须是 torch.Tensor")
    if eps_cond.shape != eps_uncond.shape:
        raise ValueError(f"形状不一致：{tuple(eps_cond.shape)} vs {tuple(eps_uncond.shape)}")
    if eps_cond.dim() < 2:
        raise ValueError("至少需要二维输入（batch × 特征）")

    w = float(guidance_scale)
    if not math.isfinite(w) or w < 0.0:
        raise ValueError("guidance_scale 必须是非负有限实数")
    phi = float(rescale)
    if not math.isfinite(phi) or not (0.0 <= phi <= 1.0):
        raise ValueError("rescale 必须落在 [0, 1] 内")
    cur, lo, hi = float(t), float(t_low), float(t_high)
    if not (math.isfinite(cur) and math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError("t / t_low / t_high 必须是有限实数")
    if lo > hi:
        raise ValueError("要求 t_low <= t_high")
    if not (0.0 <= lo and hi <= 1.0):
        raise ValueError("t_low / t_high 必须落在 [0, 1] 内")
    if not (0.0 <= cur <= 1.0):
        raise ValueError("t 必须落在 [0, 1] 内")

    red = _std_dims(dims, eps_cond.dim())

    cond = eps_cond.to(torch.float64)
    uncond = eps_uncond.to(torch.float64)
    if not (lo <= cur <= hi):
        return cond.to(torch.float32)

    out = uncond + w * (cond - uncond)
    if phi > 0.0:
        std_cond = cond.std(dim=list(red), unbiased=False)
        std_out = out.std(dim=list(red), unbiased=False)
        if bool((std_out == 0.0).any()):
            raise ValueError("引导后预测的标准差为 0，无法做 rescale")
        ratio = std_cond / std_out
        # 把按 (batch, 保留维...) 排列的比值铺回完整形状，被 reduce 的维度补 1
        broadcast = [1] * eps_cond.dim()
        for d in range(eps_cond.dim()):
            if d not in red:
                broadcast[d] = eps_cond.shape[d]
        out = (1.0 - phi) * out + phi * (out * ratio.reshape(broadcast))

    return out.to(torch.float32)
