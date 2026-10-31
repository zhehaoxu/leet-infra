"""参考解：QAQ 误差感知 KV 位宽反解（QAQ/src/quantizer.py）。

L59-64 level->dims | L198-212 kthvalue 双阈值 + (<= / >) | L159-172 两条 max_error
L176-195 log2 -> ceil -> clamp -> 末行覆盖 | L215-262 归一化 + 两个量化器 | L269-284 写回
"""

import math

import torch

_DIMS = {"token": (-3, -2, -1), "layer": (-2, -1), "head": (-1,)}


def _norm(x, mask, dims, sym, std, n):
    """L215-241：outlier 之外求中心与尺度；std 用 ddof=1，且始终围绕真实均值。"""
    keep = ~mask
    cnt = keep.sum(dims)
    if int(cnt.min()) < (2 if std else 1):
        raise ValueError("每个量化单元的非 outlier 元素太少")
    kc = cnt.reshape(cnt.shape + (1,) * len(dims))
    zero = torch.zeros((), dtype=x.dtype)
    mu = torch.where(keep, x, zero).sum(dims, keepdim=True) / kc
    mean = zero if sym else mu
    if std:
        dev = torch.where(keep, x - mu, zero)
        return mean, torch.sqrt((dev * dev).sum(dims, keepdim=True) / (kc - 1))
    if sym:
        return mean, 2 * torch.where(keep, x.abs(), zero).amax(dims, keepdim=True) / 2.0 ** n
    hi = torch.where(keep, x, torch.full((), -math.inf, dtype=x.dtype)).amax(dims, keepdim=True)
    lo = torch.where(keep, x, torch.full((), math.inf, dtype=x.dtype)).amin(dims, keepdim=True)
    return mean, (hi - lo) / 2.0 ** n


def _uniform(x, mask, dims, sym, n):
    mean, scale = _norm(x, mask, dims, sym, False, n)
    q = torch.round((x - mean) / scale).clamp(-(2.0 ** (n - 1)), 2.0 ** (n - 1) - 1.0)
    return torch.where(mask, x, q * scale + mean)


def _normal(x, mask, dims, sym, n, upper, mid):
    mean, scale = _norm(x, mask, dims, sym, True, n)
    idx = torch.searchsorted(upper, ((x - mean) / scale).contiguous())
    return torch.where(mask, x, mid[idx] * scale + mean)


def solve(cache, attentions, key_or_value, level, symmetric, outliers_ratio,
          last_n_attentions, target_quantization_error, n_bits_min, n_bits_max,
          q_norm=None, method="uniform", quantile_upper=None, quantile_center=None):
    if cache.dim() != 5:
        raise ValueError("cache 必须是五维 (n_batch, seq_len, n_layer, n_head, head_dim)")
    if cache.dtype not in (torch.float32, torch.float64):
        raise ValueError("cache 必须是 float32/float64")
    if key_or_value not in ("key", "value"):
        raise ValueError("key_or_value 只能是 'key' 或 'value'")
    if level not in _DIMS:
        raise ValueError("level 只能是 token/layer/head")
    if method not in ("uniform", "normal"):
        raise ValueError("method 只能是 uniform 或 normal")
    if not isinstance(symmetric, bool):
        raise ValueError("symmetric 必须是 bool")
    if isinstance(outliers_ratio, bool) or not isinstance(outliers_ratio, (int, float)) \
            or not 0.0 <= outliers_ratio < 1.0:
        raise ValueError("outliers_ratio 必须落在 [0, 1)")
    for name, val, lo, hi in (("last_n_attentions", last_n_attentions, 1, None),
                              ("n_bits_min", n_bits_min, 0, 16), ("n_bits_max", n_bits_max, 0, 16)):
        if isinstance(val, bool) or not isinstance(val, int) or val < lo or (hi and val > hi):
            raise ValueError(f"{name} 越界")
    if n_bits_min > n_bits_max or not target_quantization_error > 0.0:
        raise ValueError("位宽区间或目标误差非法")

    seq_len, dims = int(cache.shape[1]), _DIMS[level]
    x, nq = cache.to(torch.float64), len(dims)
    lead = tuple(cache.shape[:-nq])
    flat_len = 1
    for size in cache.shape[-nq:]:
        flat_len *= int(size)

    if outliers_ratio == 0.0:                        # L198-212：双 kthvalue + 非对称比较
        mask = torch.zeros_like(x, dtype=torch.bool)
    else:
        flat = x.reshape(*lead, flat_len)
        low = torch.kthvalue(flat, int(math.ceil(outliers_ratio / 2 * flat_len)), -1).values
        high = torch.kthvalue(flat, int(math.floor((1 - outliers_ratio / 2) * flat_len)), -1).values
        mask = (x <= low.reshape(*lead, *([1] * nq))) | (x > high.reshape(*lead, *([1] * nq)))

    if key_or_value == "key":                        # L159-172：两条 max_error
        if q_norm is None or q_norm <= 0.0:
            raise ValueError("key 分支必须给出正的 q_norm")
        if seq_len < 2:
            raise ValueError("key 分支要求 seq_len >= 2")
        err = torch.full(lead, math.sqrt(12.0 / q_norm * math.log(
            seq_len ** 3 / (seq_len - 1) * target_quantization_error ** 2 + 1)), dtype=cache.dtype)
    else:
        if attentions is None or attentions.dim() != 5:
            raise ValueError("value 分支必须给出五维 attentions")
        if tuple(attentions.shape[:3]) != (cache.shape[2], cache.shape[0], cache.shape[3]):
            raise ValueError("attentions 的 layer/batch/head 与 cache 不一致")
        if int(attentions.shape[4]) != seq_len or int(attentions.shape[3]) < last_n_attentions:
            raise ValueError("attentions 的形状与 cache / last_n_attentions 不符")
        a = attentions.to(torch.float64)[:, :, :, -last_n_attentions:]
        err = math.sqrt(12.0 / seq_len) * target_quantization_error / a.permute(1, 4, 0, 2, 3).amax(dims)

    keep = ~mask
    if int(keep.sum(dims).min()) < 1:
        raise ValueError("每个量化单元至少要有 1 个非 outlier 元素")
    zero = torch.zeros((), dtype=x.dtype)
    if symmetric:
        span = 2 * torch.where(keep, x.abs(), zero).amax(dims)
    else:
        span = (torch.where(keep, x, torch.full((), -math.inf, dtype=x.dtype)).amax(dims)
                - torch.where(keep, x, torch.full((), math.inf, dtype=x.dtype)).amin(dims))
    clamped = (span / (2 * err.to(x.dtype)) + 1.0).log2().ceil().to(torch.int64)
    clamped = clamped.clamp(n_bits_min, n_bits_max)  # L191-195：先 ceil 再 clamp
    n_bits = clamped.clone()
    if last_n_attentions > 1:
        n_bits[:, -last_n_attentions + 1:] = n_bits_max

    if method == "normal":
        if quantile_upper is None or quantile_center is None:
            raise ValueError("normal 必须给出量化表")
        if len(quantile_upper) <= n_bits_max or len(quantile_center) <= n_bits_max:
            raise ValueError("量化表必须覆盖到 n_bits_max")
        for n in range(n_bits_min, n_bits_max + 1):
            if len(quantile_upper[n]) != 2 ** n or len(quantile_center[n]) != 2 ** n:
                raise ValueError(f"第 {n} 位的量化表长度应为 2**{n}")

    out = x.clone()                                  # L269-284：按位宽分组逐组量化
    for n in range(int(n_bits.min()), int(n_bits.max()) + 1):
        sel = n_bits == n
        if not bool(sel.any()):
            continue
        if method == "uniform":
            out[sel] = _uniform(x[sel], mask[sel], dims, symmetric, n)
        else:
            out[sel] = _normal(x[sel], mask[sel], dims, symmetric, n,
                               quantile_upper[n].to(torch.float64),
                               quantile_center[n].to(torch.float64))

    return {"outlier_mask": mask, "max_error": err.to(cache.dtype),
            "n_bits_clamped": clamped, "n_bits": n_bits, "quant_dequant": out.to(cache.dtype)}
