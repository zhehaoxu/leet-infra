import math

import torch

_FLOAT_DTYPES = (torch.float32, torch.float64)


def _check_tensor(name, value, min_ndim):
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if value.dtype not in _FLOAT_DTYPES:
        raise ValueError(f"{name} must be float32 or float64")
    if value.ndim < min_ndim:
        raise ValueError(f"{name} must have at least {min_ndim} dimensions")
    if value.shape[-1] < 1:
        raise ValueError(f"{name} must have a non-empty head dimension")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


def _check_eps(eps):
    if isinstance(eps, bool) or not isinstance(eps, (int, float)):
        raise ValueError("eps must be a real number")
    eps = float(eps)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and > 0")
    return eps


def solve(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
    qk_norm: bool = False,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Squared-ReLU attention：relu(logits)² 替 softmax，再按行归一。

    query (..., L, hd)、key (..., S, hd)、value (..., S, vd)，前置维按广播规则对齐
    mask : 可广播到 (...broadcast..., L, S) 的 bool 张量，True = 保留、False = 置 0
    qk_norm : True 时先对 q、k 在 head 维做无权 RMSNorm（q / sqrt(mean(q²) + eps)）
    eps : qk_norm 的 RMS eps，同时是行归一化分母的下界

    计算口径（内部 float64）：
      1. qk_norm 为 True 时 q、k 各自按 head 维做 RMSNorm（无仿射）
      2. logits = q · kᵀ / sqrt(hd)
      3. a = relu(logits)²（mask 位置再置 0）
      4. denom = a.sum(-1)；weights = a / max(denom, eps)
      5. output = weights · value
    返回 (output, weights)：output 形状 (..., L, vd)、weights 形状 (..., L, S)，均 float32。

    性质提示：relu² 是二次齐次、且行归一会把任何正的常数温度精确抵消，
    所以 1/sqrt(hd) 不改变 weights；denom 为 0（整行无正 logit 或整行被 mask）时
    该行 weights 全 0、输出为 0；0 < denom < eps 时整行被等比缩小，行和 < 1。
    """
    _check_tensor("query", query, 2)
    _check_tensor("key", key, 2)
    _check_tensor("value", value, 2)
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key must share the head dimension")
    if tuple(key.shape[:-1]) != tuple(value.shape[:-1]):
        raise ValueError("key and value must share all leading dimensions")
    if not isinstance(qk_norm, bool):
        raise ValueError("qk_norm must be a bool")
    eps = _check_eps(eps)
    try:
        leading = torch.broadcast_shapes(query.shape[:-2], key.shape[:-2])
        torch.broadcast_shapes(leading, value.shape[:-2])
    except RuntimeError as exc:
        raise ValueError("batch dimensions are not broadcastable") from exc

    q = query.to(torch.float64)
    k = key.to(torch.float64)
    v = value.to(torch.float64)
    if qk_norm:
        q = q * torch.rsqrt(q.pow(2).mean(dim=-1, keepdim=True) + eps)
        k = k * torch.rsqrt(k.pow(2).mean(dim=-1, keepdim=True) + eps)
    logits = (q @ k.transpose(-1, -2)) / math.sqrt(query.shape[-1])
    attn = torch.relu(logits).pow(2)

    if mask is not None:
        if not isinstance(mask, torch.Tensor):
            raise ValueError("mask must be a bool torch.Tensor or None")
        if mask.dtype != torch.bool:
            raise ValueError("mask must be bool (True = keep)")
        try:
            keep = torch.broadcast_to(mask, attn.shape)
        except RuntimeError as exc:
            raise ValueError("mask is not broadcastable to the score matrix") from exc
        attn = attn.masked_fill(~keep, 0.0)

    denom = attn.sum(dim=-1, keepdim=True)
    weights = attn / denom.clamp_min(eps)
    output = weights @ v
    return output.to(torch.float32), weights.to(torch.float32)
