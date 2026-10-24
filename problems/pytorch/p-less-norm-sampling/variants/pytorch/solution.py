"""参考解：P-less-norm 采样（免超参；随机性以显式入参 u 注入，保证判题确定性）。

流程（逐行）：
1. probs = softmax(logits)。
2. v = probs.shape[-1]；thr = (v * (probs**2).sum(-1, keepdim=True) - 1) / (v - 1)。
   （把碰撞概率 Σp² ∈ [1/v, 1] 仿射归一到 [0, 1]，比原版 P-less 宽松）
3. 保留 probs >= thr，其余置 0（可证 p_max >= thr 恒成立，至少留一个）。
4. 保留集合重新归一化 → filtered（形状同 probs，和为 1）。
5. 用 u∈[0,1) 在 filtered 的累积分布（原顺序）上做逆变换采样：首个 cumsum >= u 的位置。

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


def solve(logits, u):
    probs = torch.softmax(logits, dim=-1)
    v = probs.shape[-1]
    thr = (v * probs.pow(2).sum(dim=-1, keepdim=True) - 1.0) / (v - 1.0)
    keep = (probs >= thr).to(probs.dtype)
    filtered = probs * keep
    filtered = filtered / filtered.sum(dim=-1, keepdim=True)
    return filtered, _inverse_cdf(filtered, u)
