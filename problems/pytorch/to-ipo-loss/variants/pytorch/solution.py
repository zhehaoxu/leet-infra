"""参考答案：IPO 恒等偏好优化的平方目标。"""

import torch


def _check(logp_w, logp_l, ref_logp_w, ref_logp_l, beta):
    for name, t in (("logp_w", logp_w), ("logp_l", logp_l),
                    ("ref_logp_w", ref_logp_w), ("ref_logp_l", ref_logp_l)):
        if not isinstance(t, torch.Tensor):
            raise ValueError(f"{name} 必须是张量")
        if t.dim() != 1:
            raise ValueError(f"{name} 必须是一维")
        if t.dtype not in (torch.float32, torch.float64):
            raise ValueError(f"{name} 必须是浮点")
        if not bool(torch.isfinite(t).all()):
            raise ValueError(f"{name} 含 NaN/Inf")
        if t.shape != logp_w.shape:
            raise ValueError(f"{name} 形状必须一致")
        if t.dtype != logp_w.dtype:
            raise ValueError(f"{name} dtype 必须一致")
    if logp_w.numel() == 0:
        raise ValueError("样本数必须 >= 1")
    if not isinstance(beta, (int, float)) or not beta > 0:
        raise ValueError("beta 必须 > 0")


def solve(logp_w, logp_l, ref_logp_w, ref_logp_l, beta):
    _check(logp_w, logp_l, ref_logp_w, ref_logp_l, beta)
    f64 = torch.float64
    delta = ((logp_w.to(f64) - ref_logp_w.to(f64))
             - (logp_l.to(f64) - ref_logp_l.to(f64)))
    target = 1.0 / (2.0 * float(beta))
    return ((delta - target) ** 2).mean()
