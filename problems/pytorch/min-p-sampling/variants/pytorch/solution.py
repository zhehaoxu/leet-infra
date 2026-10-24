"""参考答案：Min-p 采样（随机性以显式入参 u 注入，保证判题确定性）。

流程：
1. probs = softmax(logits)。
2. p_max = probs.max(-1)；保留 probs >= p * p_max 的候选（最高者必留，故至少 1 个）。
   —— 因 softmax 单调，该判据等价于 logit >= logit_max + ln(p)。
3. 保留集合重新归一化 → filtered。
4. 用 u 在 filtered（原顺序）的 CDF 上逆变换采样 → chosen 原始索引。

返回 (filtered, chosen)。
"""

import torch


def _inverse_cdf(filtered, u):
    if u.dim() < filtered.dim():
        u = u.view(u.shape + (1,) * (filtered.dim() - u.dim()))
    cdf = torch.cumsum(filtered, dim=-1)
    ge = cdf >= u
    pos = ge.to(torch.long).argmax(dim=-1)
    pos = torch.where(ge.any(dim=-1), pos, torch.full_like(pos, filtered.shape[-1] - 1))
    return pos


def solve(logits, p, u):
    probs = torch.softmax(logits, dim=-1)
    p_max = probs.max(dim=-1, keepdim=True).values
    keep = (probs >= float(p) * p_max).to(probs.dtype)
    filtered = probs * keep
    filtered = filtered / filtered.sum(dim=-1, keepdim=True)
    return filtered, _inverse_cdf(filtered, u)
