"""参考答案：Dr. GRPO 与 REINFORCE++ 的归一化账。"""

import torch


def _check(rewards, logp, logp_old, mask, eps):
    for name, t in (("rewards", rewards), ("logp", logp), ("logp_old", logp_old)):
        if not isinstance(t, torch.Tensor):
            raise ValueError(f"{name} 必须是张量")
        if t.dtype not in (torch.float32, torch.float64):
            raise ValueError(f"{name} 必须是浮点")
        if not bool(torch.isfinite(t).all()):
            raise ValueError(f"{name} 含 NaN/Inf")
    if rewards.dim() != 2:
        raise ValueError("rewards 必须是二维 (B,G)")
    if logp.dim() != 3:
        raise ValueError("logp 必须是三维 (B,G,T)")
    b, g, t_len = logp.shape
    if rewards.shape != (b, g):
        raise ValueError("rewards 形状必须是 (B,G)")
    if logp_old.shape != logp.shape:
        raise ValueError("logp_old 形状必须与 logp 一致")
    if g < 2:
        raise ValueError("组大小 G 必须 >= 2")
    if t_len < 1:
        raise ValueError("序列宽度 T 必须 >= 1")
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        raise ValueError("mask 必须是 bool 张量")
    if mask.shape != logp.shape:
        raise ValueError("mask 形状必须与 logp 一致")
    if bool((mask.sum(-1) < 1).any()):
        raise ValueError("每条回答至少要有一个有效 token")
    if not isinstance(eps, (int, float)) or not 0.0 < eps < 1.0:
        raise ValueError("eps 必须落在 (0, 1)")


def solve(rewards, logp, logp_old, mask, eps=0.2):
    _check(rewards, logp, logp_old, mask, eps)
    f64 = torch.float64
    r = rewards.to(f64)
    b, g, t_len = logp.shape
    adv_dr = r - r.mean(dim=1, keepdim=True)
    mu = r.mean()
    sigma = r.std(unbiased=False)
    adv_rp = torch.zeros_like(r) if float(sigma) <= 1e-12 else (r - mu) / sigma
    rho = torch.exp(logp.to(f64) - logp_old.to(f64))
    clipped = rho.clamp(1.0 - float(eps), 1.0 + float(eps))

    def surrogate(adv):
        a = adv.unsqueeze(-1)
        return torch.minimum(rho * a, clipped * a)

    token_dr = (surrogate(adv_dr) * mask).sum()
    loss_dr = -token_dr / (b * g * t_len)
    seq_rp = (surrogate(adv_rp) * mask).sum(-1) / mask.sum(-1)
    loss_rp = -seq_rp.mean()
    return {
        "adv_drgrpo": adv_dr,
        "adv_reinforcepp": adv_rp,
        "loss_drgrpo": loss_dr,
        "loss_reinforcepp": loss_rp,
    }
