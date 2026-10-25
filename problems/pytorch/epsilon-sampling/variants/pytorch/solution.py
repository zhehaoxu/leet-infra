"""参考答案：Epsilon 采样（随机性以显式入参 u 注入，保证判题确定性）。

流程：
1. probs = softmax(logits)。
2. 逐行阈值 thr = eps * probs.max(-1)；**剔除 probs < thr** 的候选（判据用 >=，等于阈值保留，
   最高概率 token 恒保留，故 filtered 分母恒正）。
3. 两种 renorm 口径：
   - renorm=True（传统截断）：保留集合重新归一化 filtered = probs*keep / sum(kept)。
   - renorm=False（论文口径 "fill the hole"）：保留集合**不**缩放，被剔除的质量
     hole = 1 - sum(kept) 全部补给该行**原始概率最大**的 token（并列取索引最小者），
     其余保留项保持原始概率不变。
4. 用 u 在 filtered（原顺序）的 CDF 上做逆变换采样 → chosen 原始索引。

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


def solve(logits, eps, renorm, u):
    probs = torch.softmax(logits, dim=-1)
    p_max = probs.max(dim=-1, keepdim=True).values
    keep = (probs >= float(eps) * p_max).to(probs.dtype)
    kept = probs * keep
    if bool(renorm):
        filtered = kept / kept.sum(dim=-1, keepdim=True)
    else:
        filtered = kept.clone()
        hole = 1.0 - kept.sum(dim=-1, keepdim=True)
        top = probs.argmax(dim=-1, keepdim=True)
        filtered.scatter_add_(-1, top, hole)
    return filtered, _inverse_cdf(filtered, u)
