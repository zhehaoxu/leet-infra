"""参考答案：CISPO 停梯度裁剪权重 + K3 KL 估计。"""

import torch


def _check(logp, logp_old, ref_logp, mask, adv, eps_low, eps_high):
    for name, t in (("logp", logp), ("logp_old", logp_old),
                    ("ref_logp", ref_logp), ("adv", adv)):
        if not isinstance(t, torch.Tensor):
            raise ValueError(f"{name} 必须是张量")
        if t.dtype not in (torch.float32, torch.float64):
            raise ValueError(f"{name} 必须是浮点")
        if not bool(torch.isfinite(t).all()):
            raise ValueError(f"{name} 含 NaN/Inf")
    if logp.dim() != 2:
        raise ValueError("logp 必须是二维 (N,T)")
    n, t_len = logp.shape
    if logp_old.shape != logp.shape or ref_logp.shape != logp.shape:
        raise ValueError("logp / logp_old / ref_logp 形状必须一致")
    if n < 1 or t_len < 1:
        raise ValueError("N 与 T 必须 >= 1")
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        raise ValueError("mask 必须是 bool 张量")
    if mask.shape != logp.shape:
        raise ValueError("mask 形状必须与 logp 一致")
    if int(mask.sum()) == 0:
        raise ValueError("至少要有一个有效 token")
    if adv.shape != (n,) or adv.dtype != logp.dtype:
        raise ValueError("adv 必须是 (N,) 且 dtype 与 logp 一致")
    if not isinstance(eps_low, (int, float)) or not 0.0 <= eps_low < 1.0:
        raise ValueError("eps_low 必须落在 [0, 1)")
    if not isinstance(eps_high, (int, float)) or eps_high < 0:
        raise ValueError("eps_high 必须 >= 0")
    log_r = ref_logp.to(torch.float64) - logp.to(torch.float64)
    if float(log_r.detach().max()) > 20.0:
        raise ValueError("ref_logp - logp 超过 20，K3 的 exp 会溢出")


def solve(logp, logp_old, ref_logp, mask, adv, eps_low=0.2, eps_high=0.2):
    _check(logp, logp_old, ref_logp, mask, adv, eps_low, eps_high)
    f64 = torch.float64
    lp = logp.to(f64)
    rho = torch.exp(lp - logp_old.to(f64))
    weight = rho.detach().clamp(1.0 - float(eps_low), 1.0 + float(eps_high))
    n = mask.sum()
    loss = -(weight * adv.to(f64).unsqueeze(-1) * lp * mask).sum() / n
    log_r = ref_logp.to(f64) - lp
    k3 = torch.exp(log_r) - log_r - 1.0
    kl_k3 = (k3 * mask).sum() / n
    outside = (rho < 1.0 - float(eps_low)) | (rho > 1.0 + float(eps_high))
    clip_frac = outside[mask].to(f64).mean()
    return {"loss": loss, "kl_k3": kl_k3, "clip_frac": clip_frac}
