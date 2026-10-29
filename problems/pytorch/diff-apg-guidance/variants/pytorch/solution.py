import math

import torch

_NORM_EPS = 1e-12


def _reduce_dims(dims, ndim):
    if dims is None:
        dims = tuple(range(1, ndim))
    else:
        if isinstance(dims, int) and not isinstance(dims, bool):
            dims = (dims,)
        dims = tuple(dims)
    if not dims:
        raise ValueError("dims 不能为空：至少需要一个归约维度")
    for d in dims:
        if isinstance(d, bool) or not isinstance(d, int):
            raise ValueError("dims 只能包含整数维度")
        if d == 0:
            raise ValueError("dims 不能包含第 0 维（要为每个样本单独做投影）")
        if d < 0 or d >= ndim:
            raise ValueError(f"维度 {d} 超出范围（ndim={ndim}）")
    if len(set(dims)) != len(dims):
        raise ValueError("dims 里有重复维度")
    return dims


def solve(
    eps_cond,
    eps_uncond,
    guidance_scale,
    eta=1.0,
    norm_threshold=0.0,
    momentum=None,
    prev_momentum=None,
    dims=None,
):
    if not torch.is_tensor(eps_cond) or not torch.is_tensor(eps_uncond):
        raise ValueError("eps_cond 与 eps_uncond 必须是 torch.Tensor")
    if eps_cond.shape != eps_uncond.shape:
        raise ValueError(f"形状不一致：{tuple(eps_cond.shape)} vs {tuple(eps_uncond.shape)}")
    if eps_cond.dim() < 2:
        raise ValueError("至少需要二维输入（batch × 特征）")

    scale = float(guidance_scale)
    if not math.isfinite(scale) or scale < 0.0:
        raise ValueError("guidance_scale 必须是非负有限实数")
    keep_ratio = float(eta)
    if not math.isfinite(keep_ratio) or not (0.0 <= keep_ratio <= 1.0):
        raise ValueError("eta 必须落在 [0, 1] 内")
    threshold = float(norm_threshold)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("norm_threshold 必须是非负有限实数")
    if momentum is not None:
        mom = float(momentum)
        if not math.isfinite(mom) or not (0.0 <= mom < 1.0):
            raise ValueError("momentum 必须落在 [0, 1) 内")
    elif prev_momentum is not None:
        raise ValueError("未启用动量（momentum=None）时不应传入 prev_momentum")

    red = _reduce_dims(dims, eps_cond.dim())

    cond = eps_cond.to(torch.float64)
    uncond = eps_uncond.to(torch.float64)
    diff = cond - uncond

    buffer = None
    if momentum is not None:
        prev = 0.0
        if prev_momentum is not None:
            if not torch.is_tensor(prev_momentum):
                raise ValueError("prev_momentum 必须是 torch.Tensor")
            if prev_momentum.shape != diff.shape:
                raise ValueError(
                    f"prev_momentum 形状不一致：{tuple(prev_momentum.shape)} vs {tuple(diff.shape)}"
                )
            prev = prev_momentum.to(torch.float64)
        # 缓冲保存的是截断前的原始差分，下一步还会再乘 momentum
        buffer = diff + float(momentum) * prev
        diff = buffer

    if threshold > 0.0:
        norm = torch.linalg.vector_norm(diff, ord=2, dim=red, keepdim=True)
        diff = diff * torch.clamp(threshold / norm, max=1.0)

    unit = cond / torch.clamp(
        torch.linalg.vector_norm(cond, ord=2, dim=red, keepdim=True), min=_NORM_EPS
    )
    parallel = torch.sum(diff * unit, dim=red, keepdim=True) * unit
    update = (diff - parallel) + keep_ratio * parallel

    pred = uncond + scale * update
    new_momentum = None if buffer is None else buffer.to(torch.float32)
    return pred.to(torch.float32), new_momentum
