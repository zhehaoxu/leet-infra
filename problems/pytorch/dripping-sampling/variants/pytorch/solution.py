"""参考答案：Dripping（滴滤）采样 —— min-p 的「头尾对偶」，随机性以显式入参 u 注入。

流程：
1. probs = softmax(logits)。
2. 只用「有效项」做统计：valid = probs > 1e-35（避免下溢项污染均值/方差；这些项本来就该被砍）。
   mean = sum(valid)/n，std = sqrt(sum(p^2)/n - mean^2) 为**总体**标准差（分母 n，不是 n-1）。
3. 阈值两式（mode 字符串二选一）：
   - mode="mean"：thr = drip_amount * mean
   - mode="std" ：thr = mean + drip_amount * std
4. 剔除**头部**：cut = probs > thr（等于阈值不剔除）。由于 softmax 单调，cut 恰是概率最高的一批。
5. 兜底（与 llama.cpp 的 min_keep 口径一致）：
   - 没有任何 token 被砍 → 不触发；
   - 存活数 < min_keep → 整行不砍（宁可不砍也不能把分布砍空/砍到只剩几个尾巴）。
6. 存活集合重新归一化得 filtered（原顺序），用 u 在原顺序 CDF 上逆变换采样得 chosen。

返回 (filtered, chosen, cut_count)；cut_count 为 int64 张量 (...)，0 表示未触发。
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


def _stats(probs):
    valid = probs > 1e-35
    n = valid.sum(dim=-1, keepdim=True).clamp_min(1).to(probs.dtype)
    s = (probs * valid).sum(dim=-1, keepdim=True)
    s2 = (probs * probs * valid).sum(dim=-1, keepdim=True)
    mean = s / n
    var = (s2 / n - mean * mean).clamp_min(0.0)
    return mean, torch.sqrt(var)


def solve(logits, mode, drip_amount, min_keep, u):
    probs = torch.softmax(logits, dim=-1)
    mean, std = _stats(probs)
    amount = float(drip_amount)
    if str(mode) == "mean":
        thr = amount * mean
    elif str(mode) == "std":
        thr = mean + amount * std
    else:
        raise ValueError(f"mode 只能是 'mean' 或 'std'，收到 {mode!r}")

    cut = probs > thr
    survivors = probs.shape[-1] - cut.sum(dim=-1)
    fired = (cut.sum(dim=-1) > 0) & (survivors >= int(min_keep))
    keep = torch.where(fired.unsqueeze(-1), (~cut).to(probs.dtype), torch.ones_like(probs))

    filtered = probs * keep
    filtered = filtered / filtered.sum(dim=-1, keepdim=True)
    cut_count = torch.where(fired, cut.sum(dim=-1), torch.zeros_like(fired, dtype=torch.long))
    return filtered, _inverse_cdf(filtered, u), cut_count.to(torch.long)
