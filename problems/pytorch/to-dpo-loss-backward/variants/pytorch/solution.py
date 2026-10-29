"""参考答案：DPO 损失的解析反向。"""

import torch
import torch.nn.functional as F


def _check(logp_w, logp_l, ref_logp_w, ref_logp_l, mask_w, mask_l, beta):
    for name, t in (("logp_w", logp_w), ("logp_l", logp_l)):
        if not isinstance(t, torch.Tensor):
            raise ValueError(f"{name} 必须是张量")
        if t.dim() != 2:
            raise ValueError(f"{name} 必须是二维 (B,T)")
        if t.dtype not in (torch.float32, torch.float64):
            raise ValueError(f"{name} 必须是浮点")
        if not bool(torch.isfinite(t).all()):
            raise ValueError(f"{name} 含 NaN/Inf")
    b = logp_w.shape[0]
    if logp_l.shape != logp_w.shape:
        raise ValueError("logp_w 与 logp_l 形状必须一致")
    if b < 1 or logp_w.shape[1] < 1:
        raise ValueError("B 与 T 必须 >= 1")
    for name, t in (("ref_logp_w", ref_logp_w), ("ref_logp_l", ref_logp_l)):
        if not isinstance(t, torch.Tensor):
            raise ValueError(f"{name} 必须是张量")
        if t.shape != (b,):
            raise ValueError(f"{name} 形状必须是 (B,)")
        if t.dtype != logp_w.dtype:
            raise ValueError(f"{name} dtype 必须与 logp 一致")
        if not bool(torch.isfinite(t).all()):
            raise ValueError(f"{name} 含 NaN/Inf")
    for name, t in (("mask_w", mask_w), ("mask_l", mask_l)):
        if not isinstance(t, torch.Tensor) or t.dtype != torch.bool:
            raise ValueError(f"{name} 必须是 bool 张量")
        if t.shape != logp_w.shape:
            raise ValueError(f"{name} 形状必须与 logp 一致")
        if bool((t.sum(-1) < 1).any()):
            raise ValueError(f"{name} 每条样本至少要有 1 个有效 token")
    if not isinstance(beta, (int, float)) or not beta > 0:
        raise ValueError("beta 必须 > 0")


def solve(logp_w, logp_l, ref_logp_w, ref_logp_l, mask_w, mask_l, beta):
    _check(logp_w, logp_l, ref_logp_w, ref_logp_l, mask_w, mask_l, beta)
    f64 = torch.float64
    seq_w = (logp_w.to(f64) * mask_w).sum(-1)
    seq_l = (logp_l.to(f64) * mask_l).sum(-1)
    delta = (seq_w - ref_logp_w.to(f64)) - (seq_l - ref_logp_l.to(f64))
    x = float(beta) * delta
    loss = F.softplus(-x).mean()
    d_delta = -float(beta) * torch.sigmoid(-x) / delta.numel()
    return {
        "loss": loss,
        "grad_logp_w": d_delta.unsqueeze(-1) * mask_w,
        "grad_logp_l": -d_delta.unsqueeze(-1) * mask_l,
    }
