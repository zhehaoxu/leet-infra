"""参考答案：DAPO 损失（clip-higher + token 级聚合 + 组过滤）。"""

import torch


def _check(logp, logp_old, mask, rewards, eps_low, eps_high):
    for name, t in (("logp", logp), ("logp_old", logp_old), ("rewards", rewards)):
        if not isinstance(t, torch.Tensor):
            raise ValueError(f"{name} 必须是张量")
        if t.dtype not in (torch.float32, torch.float64):
            raise ValueError(f"{name} 必须是浮点")
        if not bool(torch.isfinite(t).all()):
            raise ValueError(f"{name} 含 NaN/Inf")
    if logp.dim() != 3:
        raise ValueError("logp 必须是三维 (B,G,T)")
    b, g, t_len = logp.shape
    if logp_old.shape != logp.shape:
        raise ValueError("logp_old 形状必须与 logp 一致")
    if g < 2 or t_len < 1:
        raise ValueError("G 必须 >= 2 且 T 必须 >= 1")
    if rewards.shape != (b, g):
        raise ValueError("rewards 形状必须是 (B,G)")
    if bool(((rewards != 0) & (rewards != 1)).any()):
        raise ValueError("rewards 必须是 0/1 二值正确性标记")
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        raise ValueError("mask 必须是 bool 张量")
    if mask.shape != logp.shape:
        raise ValueError("mask 形状必须与 logp 一致")
    if bool((mask.sum(-1) < 1).any()):
        raise ValueError("每条回答至少要有一个有效 token")
    if not isinstance(eps_low, (int, float)) or not 0.0 <= eps_low < 1.0:
        raise ValueError("eps_low 必须落在 [0, 1)")
    if not isinstance(eps_high, (int, float)) or eps_high < 0:
        raise ValueError("eps_high 必须 >= 0")


def solve(logp, logp_old, mask, rewards, eps_low=0.2, eps_high=0.28):
    _check(logp, logp_old, mask, rewards, eps_low, eps_high)
    f64 = torch.float64
    r = rewards.to(f64)
    mean_g = r.mean(dim=1, keepdim=True)
    std_g = r.std(dim=1, keepdim=True, unbiased=False)
    adv = torch.where(std_g > 0, (r - mean_g) / std_g.clamp(min=1e-12),
                      torch.zeros_like(r))
    acc = r.mean(dim=1)
    keep = (acc > 0) & (acc < 1)
    rho = torch.exp(logp.to(f64) - logp_old.to(f64))
    clipped = rho.clamp(1.0 - float(eps_low), 1.0 + float(eps_high))
    a = adv.unsqueeze(-1)
    surrogate = torch.minimum(rho * a, clipped * a)
    valid = mask & keep[:, None, None]
    num = valid.sum()
    if int(num) == 0:
        loss = torch.zeros((), dtype=f64)
    else:
        loss = -(surrogate * valid).sum() / num
    return {"loss": loss, "adv": adv, "keep": keep, "num_tokens": num}
