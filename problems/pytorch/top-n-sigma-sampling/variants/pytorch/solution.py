"""参考解：Top-n-sigma 采样（随机性以显式入参 u 注入，保证判题确定性）。

流程（对最后一维逐行）：
1. probs = softmax(logits)。
2. 逐行统计 max、mean、以及总体标准差 std = sqrt(mean((logits-mean)^2))（除以 V）。
3. 保留 logit >= max - n*std 的候选（softmax 单调，按 logit 阈值筛等价于按概率筛），其余置 0。
4. 保留集合重新归一化 → filtered（形状同 probs，和为 1）。
5. 用 u∈[0,1) 在 filtered 的累积分布（原顺序）上做逆变换采样：取首个 cumsum >= u 的位置，
   即选中 token 的原始索引；若 u 略大于末位累积值则兜底取最后一个位置。

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


def solve(logits, n, u):
    probs = torch.softmax(logits, dim=-1)
    mx = logits.max(dim=-1, keepdim=True).values
    mean = logits.mean(dim=-1, keepdim=True)
    std = logits.sub(mean).pow(2).mean(dim=-1, keepdim=True).sqrt()
    cut = mx - float(n) * std
    keep = (logits >= cut).to(probs.dtype)
    filtered = probs * keep
    filtered = filtered / filtered.sum(dim=-1, keepdim=True)
    return filtered, _inverse_cdf(filtered, u)
