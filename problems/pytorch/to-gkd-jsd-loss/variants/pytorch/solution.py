"""参考答案：GKD 在线蒸馏损失（广义 JSD + 硬标签 CE）。"""

import math

import torch
import torch.nn.functional as F


def _check(student_logits, teacher_logits, labels, lam, beta, temperature, ignore_index):
    for name, t in (("student_logits", student_logits), ("teacher_logits", teacher_logits)):
        if not isinstance(t, torch.Tensor):
            raise ValueError(f"{name} 必须是张量")
        if t.dim() != 2:
            raise ValueError(f"{name} 必须是二维 (N, V)")
        if t.dtype not in (torch.float32, torch.float64):
            raise ValueError(f"{name} 必须是浮点")
        if not bool(torch.isfinite(t).all()):
            raise ValueError(f"{name} 含 NaN/Inf")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student_logits 与 teacher_logits 形状必须一致")
    if student_logits.shape[1] < 1:
        raise ValueError("词表大小 V 必须 >= 1")
    if not isinstance(labels, torch.Tensor):
        raise ValueError("labels 必须是张量")
    if labels.dtype not in (torch.int32, torch.int64):
        raise ValueError("labels 必须是整型")
    if labels.shape != (student_logits.shape[0],):
        raise ValueError("labels 形状必须为 (N,)")
    valid = labels != ignore_index
    if not bool(valid.any()):
        raise ValueError("至少需要一个非 ignore_index 的标签")
    v = student_logits.shape[1]
    if bool(((labels[valid] < 0) | (labels[valid] >= v)).any()):
        raise ValueError("有效标签越界")
    if not isinstance(lam, (int, float)) or not 0.0 <= lam <= 1.0:
        raise ValueError("lam 必须落在 [0, 1]")
    if not isinstance(beta, (int, float)) or not 0.0 < beta < 1.0:
        raise ValueError("beta 必须严格落在 (0, 1)")
    if not isinstance(temperature, (int, float)) or not temperature > 0:
        raise ValueError("temperature 必须 > 0")


def solve(student_logits, teacher_logits, labels, lam=0.5, beta=0.5,
          temperature=1.0, ignore_index=-100):
    _check(student_logits, teacher_logits, labels, lam, beta, temperature, ignore_index)
    f64 = torch.float64
    valid = labels != ignore_index
    sl = F.log_softmax(student_logits.to(f64) / float(temperature), dim=-1)
    tl = F.log_softmax(teacher_logits.to(f64) / float(temperature), dim=-1)
    ps, pt = sl.exp(), tl.exp()
    mixture = torch.logsumexp(
        torch.stack([tl + math.log(beta), sl + math.log1p(-beta)]), dim=0)
    kl_t = (pt * (tl - mixture)).sum(-1)
    kl_s = (ps * (sl - mixture)).sum(-1)
    jsd = beta * kl_t + (1.0 - beta) * kl_s
    ce = -F.log_softmax(student_logits.to(f64), dim=-1).gather(
        1, labels.clamp(min=0).unsqueeze(1)).squeeze(1)
    return ((1.0 - lam) * jsd[valid].mean() + lam * ce[valid].mean())
