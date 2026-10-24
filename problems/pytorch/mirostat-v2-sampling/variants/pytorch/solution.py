"""参考解：Mirostat v2（把困惑度控制在目标 tau 附近的自适应截断；mu 状态显式入/出）。

流程：
1. probs = softmax(logits)。
2. 按概率降序排，截断保留前缀直到「-log2(p) > mu」第一次成立（即丢掉比 mu 更意外的尾巴）；
   若首元素就越界则至少保留 1 个（size = max(first_over, 1)）。
3. 保留集合重新归一化 → filtered（映射回原顺序，和为 1）。
4. 用 u∈[0,1) 在 filtered（原顺序）CDF 上逆变换采样得 chosen。
5. 观测困惑度 obs = -log2(filtered[chosen])；更新 mu' = mu - eta*(obs - tau)。

返回 (filtered, chosen, mu_new)。
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


def solve(logits, tau, eta, mu, u):
    probs = torch.softmax(logits, dim=-1)
    n = probs.shape[-1]
    sorted_probs, idx = torch.sort(probs, dim=-1, descending=True, stable=True)
    surp = -torch.log2(sorted_probs.clamp_min(_TINY))
    over = surp > float(mu)
    arg = over.to(torch.long).argmax(dim=-1)
    first_over = torch.where(over.any(dim=-1), arg, torch.full_like(arg, n))
    size = torch.clamp(first_over, min=1)
    rank = torch.arange(n, device=probs.device).expand_as(idx)
    keep_sorted = (rank < size.unsqueeze(-1)).to(probs.dtype)
    keep = torch.zeros_like(probs).scatter_(-1, idx, keep_sorted)
    filtered = probs * keep
    filtered = filtered / filtered.sum(dim=-1, keepdim=True)
    chosen = _inverse_cdf(filtered, u)
    sel = filtered.gather(-1, chosen.view(-1, 1)).clamp_min(_TINY) if filtered.dim() > 1 \
        else filtered[chosen].clamp_min(_TINY)
    obs = -torch.log2(sel.reshape(filtered.shape[:-1]))
    mu_new = float(mu) - float(eta) * (obs - float(tau))
    return filtered, chosen, mu_new
