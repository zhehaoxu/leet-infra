"""参考答案：cDPO 噪声偏好标签的平滑去偏损失。"""

import torch
import torch.nn.functional as F


def _check(logp_w, logp_l, ref_logp_w, ref_logp_l, beta, eps):
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
    if not isinstance(eps, (int, float)) or not 0.0 <= eps <= 0.5:
        raise ValueError("eps 必须落在 [0, 0.5]")


def solve(logp_w, logp_l, ref_logp_w, ref_logp_l, beta, eps):
    _check(logp_w, logp_l, ref_logp_w, ref_logp_l, beta, eps)
    f64 = torch.float64
    delta = ((logp_w.to(f64) - ref_logp_w.to(f64))
             - (logp_l.to(f64) - ref_logp_l.to(f64)))
    logit = float(beta) * delta
    loss = -(1.0 - float(eps)) * F.logsigmoid(logit) - float(eps) * F.logsigmoid(-logit)
    return loss.mean()
