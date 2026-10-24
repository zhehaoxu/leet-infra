"""参考答案：交叉熵损失 —— log_softmax 稳定式 + 按整数标签 gather 出 NLL。"""

import torch


def solve(logits: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    # 减每行最大值保证 exp 不上溢（数值稳定的 log-softmax）
    m = logits.max(dim=-1, keepdim=True).values
    shifted = logits - m
    logsum = shifted.exp().sum(dim=-1, keepdim=True).log()
    logp = shifted - logsum                       # (N, C) 每类 log-prob
    idx = target.view(-1, 1).long()
    nll = -logp.gather(-1, idx).squeeze(-1)       # (N,) 每个样本的 -log p[true]
    if reduction == "none":
        return nll
    if reduction == "sum":
        return nll.sum()
    if reduction == "mean":
        return nll.mean()
    raise ValueError(f"unknown reduction: {reduction!r}")
