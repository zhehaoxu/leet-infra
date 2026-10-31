import math

import torch

_TH = 1.95
_EPS = 1e-8


def solve(W, loss, R_ref, ratio):
    """BAQ：按每列损失的幂律分配位宽，再对每行动态范围做非对称量化。"""
    if not isinstance(W, torch.Tensor) or W.dim() != 2 or not W.is_floating_point():
        raise ValueError("W 必须是二维浮点张量")
    if W.shape[0] == 0 or W.shape[1] == 0:
        raise ValueError("W 的行数和列数都必须大于 0")
    if not bool(torch.isfinite(W).all()):
        raise ValueError("W 必须有限")

    if not isinstance(loss, torch.Tensor) or loss.dim() != 1 or not loss.is_floating_point():
        raise ValueError("loss 必须是一维浮点张量")
    if loss.numel() != W.shape[1]:
        raise ValueError("loss 的长度必须等于 W 的列数")
    if not bool(torch.isfinite(loss).all()) or bool((loss < 0).any()):
        raise ValueError("loss 必须有限且非负")

    if not isinstance(R_ref, int) or isinstance(R_ref, bool) or not (1 <= R_ref <= 8):
        raise ValueError("R_ref 必须是 1 到 8 的整数")
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
        raise ValueError("ratio 必须是数值")
    if not math.isfinite(float(ratio)) or float(ratio) <= 0:
        raise ValueError("ratio 必须是有限正数")

    w = W.to(torch.float64)
    l = loss.to(torch.float64)
    loss_mean = float(ratio) * l.mean()
    if float(loss_mean) <= 0:
        raise ValueError("loss 的均值必须大于 0")

    t = (l / loss_mean) * (2.0 ** (_TH * R_ref))
    t = t.clamp_min(1.0)
    levels = t.pow(1.0 / _TH).round()
    maxq = (levels - 1).to(torch.int64)
    bits = levels.log2()

    rows = w.shape[0]
    zero_scalar = torch.zeros(rows, dtype=torch.float64)
    xmax = torch.maximum(w.max(dim=1).values, zero_scalar)
    xmin = torch.minimum(w.min(dim=1).values, zero_scalar)
    dead = (xmin == 0) & (xmax == 0)
    xmin = torch.where(dead, torch.full_like(xmin, -1.0), xmin)
    xmax = torch.where(dead, torch.full_like(xmax, 1.0), xmax)

    diff = xmax - xmin
    diff = torch.where(diff == 0, diff + _EPS, diff)
    maxq_f = maxq.to(torch.float64)
    scale = diff.unsqueeze(1) / (maxq_f.unsqueeze(0) + _EPS)
    zero = torch.round(-xmin.unsqueeze(1) / scale)

    q = torch.round(w / scale).add(zero).clamp(min=0)
    q = torch.minimum(q, maxq_f.unsqueeze(0))
    dequant = scale * (q - zero)

    return {
        "bits": bits.to(torch.float32),
        "maxq": maxq,
        "scale": scale.to(torch.float32),
        "zero": zero.to(torch.int64),
        "dequant": dequant.to(torch.float32),
    }
