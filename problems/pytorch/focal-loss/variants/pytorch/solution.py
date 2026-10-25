"""参考答案：log_softmax gather 真类 -> -(1-pt)^gamma * logpt，按 reduction 归约。"""

import torch


def solve(logits, target, gamma=2.0, alpha=None, reduction="mean"):
    logp = torch.log_softmax(logits, dim=-1)
    idx = target.view(-1, 1).long()
    log_pt = logp.gather(-1, idx).squeeze(-1)
    pt = log_pt.exp()
    loss = -((1 - pt) ** gamma) * log_pt
    if alpha is not None:
        loss = alpha * loss
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    raise ValueError(f"unknown reduction: {reduction!r}")
