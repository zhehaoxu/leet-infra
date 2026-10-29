import math

import torch

_FLOAT_DTYPES = (torch.float32, torch.float64)


def _check_scores(scores):
    if not isinstance(scores, torch.Tensor):
        raise ValueError("scores must be a torch.Tensor")
    if scores.dtype not in _FLOAT_DTYPES:
        raise ValueError("scores must be float32 or float64")
    if scores.ndim < 1:
        raise ValueError("scores must have at least one dimension")
    if scores.shape[-1] < 1:
        raise ValueError("the last dimension of scores must be >= 1")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("scores must be finite; use mask to drop positions")


def _check_sink_logit(sink_logit):
    if isinstance(sink_logit, bool) or not isinstance(sink_logit, (int, float)):
        raise ValueError("sink_logit must be a real number")
    value = float(sink_logit)
    if not math.isfinite(value):
        raise ValueError("sink_logit must be finite")
    return value


def solve(
    scores: torch.Tensor,
    mask: torch.Tensor | None = None,
    sink_logit: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SoftMax1 / LLN：分母多一个 exp(sink_logit) 的 sink 归一。

    scores : (..., S) float32/float64，必须是有限值；要屏蔽的位置用 mask 表达
    mask : 可广播到 scores 的 bool 张量，True = 保留、False = 屏蔽（权重恒 0）；None 表示全保留
    sink_logit : 虚拟 sink 位置的 logit（StreamingLLM 的 zero token 即 0.0）

    计算口径（内部 float64）：
      1. r = max(max_{mask} scores, sink_logit)           ← sink 也参与取最大，防上溢
      2. e_j = exp(scores_j - r)（仅保留位，其余 0）；e_sink = exp(sink_logit - r)
      3. denom = sum_j e_j + e_sink
      4. weights_j = e_j / denom；sink_weight = e_sink / denom

    返回 (weights, sink_weight)：weights 与 scores 同形状、float32；
    sink_weight 形状为 scores.shape[:-1]、float32。
    每行恒有 weights.sum(-1) + sink_weight == 1，因此 sum(weights) = 1 - sink_weight < 1，
    缺口就是被 sink 吃掉的那部分概率质量。
    """
    _check_scores(scores)
    sink = _check_sink_logit(sink_logit)
    x = scores.to(torch.float64)

    if mask is None:
        keep = torch.ones_like(x, dtype=torch.bool)
    else:
        if not isinstance(mask, torch.Tensor):
            raise ValueError("mask must be a bool torch.Tensor or None")
        if mask.dtype != torch.bool:
            raise ValueError("mask must be bool (True = keep)")
        try:
            keep = torch.broadcast_to(mask, x.shape)
        except RuntimeError as exc:
            raise ValueError("mask is not broadcastable to scores") from exc

    row_max = torch.where(keep, x, torch.full_like(x, float("-inf")))
    row_max = row_max.max(dim=-1, keepdim=True).values.clamp_min(sink)
    safe = torch.where(keep, x, row_max)
    exp_scores = torch.where(keep, (safe - row_max).exp(), torch.zeros_like(x))
    exp_sink = (sink - row_max).exp()
    denom = exp_scores.sum(dim=-1, keepdim=True) + exp_sink

    weights = exp_scores / denom
    sink_weight = exp_sink / denom
    return weights.to(torch.float32), sink_weight.squeeze(-1).to(torch.float32)
