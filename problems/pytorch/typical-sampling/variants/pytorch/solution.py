"""参考答案：Typical / 局部典型采样（ε-sampling，Locally Typical Sampling）。

核心不是"留高概率"，而是留"信息量典型"的 token：每个 token 的自信息（surprisal）为
- ln p，分布的熵 H = Σ -p ln p 是期望 surprisal。保留 |(- ln p) - H| 最小者，按该偏差
升序累积概率，直到首次 > tau 为止（至少留 1 个）。这样既砍掉低概率长尾（surprisal 远大于
H），也能砍掉"过于自信"的 argmax（surprisal 远小于 H），把随机性留给真正 typical 的候选。

随机性以显式入参 u 注入，逆变换采样保证判题确定性。
"""

import torch

_TINY = 1e-30


def _inverse_cdf(filtered, u):
    if u.dim() < filtered.dim():
        u = u.view(u.shape + (1,) * (filtered.dim() - u.dim()))
    cdf = torch.cumsum(filtered, dim=-1)
    ge = cdf >= u
    pos = ge.to(torch.long).argmax(dim=-1)
    pos = torch.where(ge.any(dim=-1), pos, torch.full_like(pos, filtered.shape[-1] - 1))
    return pos


def solve(logits, tau, u):
    probs = torch.softmax(logits, dim=-1)
    n = probs.shape[-1]
    logp = probs.clamp_min(_TINY).log()
    ent = -(probs * logp).sum(dim=-1, keepdim=True)      # 熵 H = Σ -p ln p
    shifted = (-logp - ent).abs()                        # |surprisal - H|
    order = torch.argsort(shifted, dim=-1, stable=True)  # 升序典型性，并列按原索引
    sorted_probs = torch.gather(probs, dim=-1, index=order)
    cum = torch.cumsum(sorted_probs, dim=-1)
    over = cum > float(tau)
    cnt = torch.where(over.any(dim=-1), over.to(torch.long).argmax(dim=-1) + 1,
                      torch.full_like(over[..., 0], n))
    rank = torch.arange(n, device=probs.device).expand_as(order)
    keep_sorted = (rank < cnt.unsqueeze(-1)).to(probs.dtype)
    keep = torch.zeros_like(probs).scatter_(-1, order, keep_sorted)
    filtered = probs * keep
    filtered = filtered / filtered.sum(dim=-1, keepdim=True)
    return filtered, _inverse_cdf(filtered, u)
