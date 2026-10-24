"""参考解：Top-k 采样（随机性以显式入参 u 注入，保证判题确定性）。

流程：
1. probs = softmax(logits)。
2. 取第 k 大的 logit 作为阈值 thr，保留 logit >= thr 的候选（softmax 单调，按 logit
   选 top-k 等价于按概率选 top-k），其余置 0。
3. 保留集合重新归一化 → filtered（形状同 probs，和为 1）。
4. 用 u∈[0,1) 在 filtered 的累积分布（原顺序）上做逆变换采样：取首个 cumsum >= u 的位置，
   即选中 token 的原始索引；若 u 略大于末位累积值则兜底取最后一个保留位置。

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


def solve(logits, k, u):
    probs = torch.softmax(logits, dim=-1)
    kk = max(1, min(int(k), logits.shape[-1]))
    thr = logits.topk(kk, dim=-1).values[..., -1:]
    keep = (logits >= thr).to(probs.dtype)
    filtered = probs * keep
    filtered = filtered / filtered.sum(dim=-1, keepdim=True)
    return filtered, _inverse_cdf(filtered, u)
