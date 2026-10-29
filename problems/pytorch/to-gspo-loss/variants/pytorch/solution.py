"""参考答案：GSPO 序列级重要性比损失。"""

import torch


def _check(logp, logp_old, mask, adv, eps):
    for name, t in (("logp", logp), ("logp_old", logp_old), ("adv", adv)):
        if not isinstance(t, torch.Tensor):
            raise ValueError(f"{name} 必须是张量")
        if t.dtype not in (torch.float32, torch.float64):
            raise ValueError(f"{name} 必须是浮点")
        if not bool(torch.isfinite(t).all()):
            raise ValueError(f"{name} 含 NaN/Inf")
    if logp.dim() != 2:
        raise ValueError("logp 必须是二维 (N,T)")
    n, t_len = logp.shape
    if logp_old.shape != logp.shape:
        raise ValueError("logp_old 形状必须与 logp 一致")
    if n < 1 or t_len < 1:
        raise ValueError("N 与 T 必须 >= 1")
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        raise ValueError("mask 必须是 bool 张量")
    if mask.shape != logp.shape:
        raise ValueError("mask 形状必须与 logp 一致")
    if bool((mask.sum(-1) < 1).any()):
        raise ValueError("每条回答至少要有一个有效 token")
    if adv.shape != (n,):
        raise ValueError("adv 形状必须是 (N,)")
    if adv.dtype != logp.dtype:
        raise ValueError("adv dtype 必须与 logp 一致")
    if not isinstance(eps, (int, float)) or not 0.0 < eps < 1.0:
        raise ValueError("eps 必须落在 (0, 1)")


def solve(logp, logp_old, mask, adv, eps=3e-4):
    _check(logp, logp_old, mask, adv, eps)
    f64 = torch.float64
    delta = (logp.to(f64) - logp_old.to(f64)) * mask
    lengths = mask.sum(-1)
    log_ratio = delta.sum(-1) / lengths
    s = torch.exp(log_ratio)
    a = adv.to(f64)
    obj = torch.minimum(s * a, s.clamp(1.0 - float(eps), 1.0 + float(eps)) * a)
    return {"loss": -obj.mean(), "ratio": s}
