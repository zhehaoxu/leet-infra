"""参考答案：F.kl_div 等价实现（input 是 log q，target 是 p 或 log p）。

p=0 处必须显式取 0：朴素写 t * (t.log() - q) 在 t=0 时是 0·(−inf)=NaN，
torch 用 0·log 0 := 0 的约定跳过这些位置；log_target=True 时 p=0 表现为 target=−inf，
同样要按符号无穷来判定跳过。剩下的 p>0、q=0 才真正给 +inf。
（注：F.kl_div 在 p=0 且 q=0 的实现细节给 NaN，本题按题面约定取 0。）
"""

import torch


def solve(log_pred, target, reduction="batchmean", log_target=False):
    if reduction not in ("none", "mean", "sum", "batchmean"):
        raise ValueError(f"未知 reduction: {reduction!r}")
    q = log_pred.double()
    t = target.double()

    if log_target:
        zero_mask = torch.isinf(t) & (t < 0)          # log p = -inf 即 p = 0
        elem = torch.where(zero_mask, torch.zeros_like(t), t.exp() * (t - q))
    else:
        elem = torch.where(t > 0, t * (t.log() - q), torch.zeros_like(t))

    if reduction == "none":
        return elem.to(log_pred.dtype)
    total = elem.sum()
    if reduction == "sum":
        return total.to(log_pred.dtype)
    if reduction == "mean":
        return (total / elem.numel()).to(log_pred.dtype)
    if elem.dim() == 0:
        raise ValueError("batchmean 要求至少一维输入")
    return (total / elem.shape[0]).to(log_pred.dtype)
