"""参考答案：ORPO 赔率偏好损失（SFT + λ·赔率项）。"""

import torch
import torch.nn.functional as F


def _check(logp_w, logp_l, len_w, len_l, lam):
    for name, t in (("logp_w", logp_w), ("logp_l", logp_l)):
        if not isinstance(t, torch.Tensor):
            raise ValueError(f"{name} 必须是张量")
        if t.dim() != 1:
            raise ValueError(f"{name} 必须是一维")
        if t.dtype not in (torch.float32, torch.float64):
            raise ValueError(f"{name} 必须是浮点")
        if not bool(torch.isfinite(t).all()):
            raise ValueError(f"{name} 含 NaN/Inf")
    if logp_w.shape != logp_l.shape:
        raise ValueError("logp_w 与 logp_l 形状必须一致")
    if logp_w.numel() == 0:
        raise ValueError("样本数必须 >= 1")
    for name, t in (("len_w", len_w), ("len_l", len_l)):
        if not isinstance(t, torch.Tensor):
            raise ValueError(f"{name} 必须是张量")
        if t.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"{name} 必须是整型")
        if t.shape != logp_w.shape:
            raise ValueError(f"{name} 形状必须与 logp 一致")
        if bool((t < 1).any()):
            raise ValueError(f"{name} 必须 >= 1")
    if logp_w.dtype != logp_l.dtype or len_w.dtype != len_l.dtype:
        raise ValueError("logp_w/logp_l 或 len_w/len_l dtype 不一致")
    if not isinstance(lam, (int, float)) or lam < 0:
        raise ValueError("lam 必须 >= 0")


def solve(logp_w, logp_l, len_w, len_l, lam=0.1):
    _check(logp_w, logp_l, len_w, len_l, lam)
    f64 = torch.float64
    lw = logp_w.to(f64) / len_w.to(f64)
    ll = logp_l.to(f64) / len_l.to(f64)
    if bool((lw >= 0).any()) or bool((ll >= 0).any()):
        raise ValueError("长度归一后的平均对数概率必须 < 0（否则 odds 无定义）")
    log_odds_w = lw - torch.log1p(-torch.exp(lw))
    log_odds_l = ll - torch.log1p(-torch.exp(ll))
    sft = -lw.mean()
    or_loss = -F.logsigmoid(log_odds_w - log_odds_l).mean()
    return sft + lam * or_loss
