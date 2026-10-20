"""参考答案：数值稳定 masked softmax。"""

import torch


def solve(logits: torch.Tensor, mask=None, dim: int = -1) -> torch.Tensor:
    x = logits
    if mask is None:
        m = x.amax(dim=dim, keepdim=True)
        e = torch.exp(x - m)
        return e / e.sum(dim=dim, keepdim=True)

    neg_inf = torch.full_like(x, float("-inf"))
    m = torch.where(mask, x, neg_inf).amax(dim=dim, keepdim=True)
    all_masked = torch.isinf(m) & (m < 0)  # 整组被 mask
    m = torch.where(all_masked, torch.zeros_like(m), m)
    # 无效位置先填 -inf 再 exp，避免 where 两个分支都计算导致的溢出
    e = torch.exp(torch.where(mask, x - m, neg_inf))
    s = e.sum(dim=dim, keepdim=True)
    out = e / s
    return torch.where(all_masked, torch.zeros_like(out), out)
