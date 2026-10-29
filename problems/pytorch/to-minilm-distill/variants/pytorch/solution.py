"""参考答案：MiniLM 注意力分布 KL + CLS 对 MSE。"""

import torch
import torch.nn.functional as F


def _check(teacher_attn, student_attn, teacher_cls, student_cls, mask, temperature):
    for name, t in (("teacher_attn", teacher_attn), ("student_attn", student_attn)):
        if not isinstance(t, torch.Tensor):
            raise ValueError(f"{name} 必须是张量")
        if t.dim() != 4:
            raise ValueError(f"{name} 必须是四维 (B,H,L,L)")
        if t.dtype not in (torch.float32, torch.float64):
            raise ValueError(f"{name} 必须是浮点")
        if not bool(torch.isfinite(t).all()):
            raise ValueError(f"{name} 含 NaN/Inf")
    if teacher_attn.shape != student_attn.shape:
        raise ValueError("teacher_attn 与 student_attn 形状必须一致")
    b, h, lq, lk = teacher_attn.shape
    if lq != lk:
        raise ValueError("注意力必须是方阵 (B,H,L,L)")
    if lq < 1 or h < 1:
        raise ValueError("L 与 H 必须 >= 1")
    if not isinstance(mask, torch.Tensor):
        raise ValueError("mask 必须是张量")
    if mask.dtype != torch.bool:
        raise ValueError("mask 必须是 bool")
    if mask.shape != (b, lq):
        raise ValueError("mask 形状必须是 (B,L)")
    if not bool(mask.any()):
        raise ValueError("mask 至少要有一个 True")
    for name, t in (("teacher_cls", teacher_cls), ("student_cls", student_cls)):
        if not isinstance(t, torch.Tensor):
            raise ValueError(f"{name} 必须是张量")
        if t.dim() != 3 or t.shape[0] != b or t.shape[1] != 3:
            raise ValueError(f"{name} 形状必须是 (B,3,D)")
        if t.dtype not in (torch.float32, torch.float64):
            raise ValueError(f"{name} 必须是浮点")
        if not bool(torch.isfinite(t).all()):
            raise ValueError(f"{name} 含 NaN/Inf")
    if teacher_cls.shape != student_cls.shape:
        raise ValueError("teacher_cls 与 student_cls 形状必须一致")
    if teacher_cls.shape[2] < 1:
        raise ValueError("D 必须 >= 1")
    if not isinstance(temperature, (int, float)) or not temperature > 0:
        raise ValueError("temperature 必须 > 0")


def solve(teacher_attn, student_attn, teacher_cls, student_cls, mask,
          temperature=1.0):
    _check(teacher_attn, student_attn, teacher_cls, student_cls, mask, temperature)
    f64 = torch.float64
    b, h, lq, _ = teacher_attn.shape
    key_mask = mask[:, None, None, :]
    q_mask = mask[:, None, :].to(f64)
    neg_inf = float("-inf")
    lt = torch.log_softmax(
        (teacher_attn.to(f64) / float(temperature)).masked_fill(~key_mask, neg_inf), dim=-1)
    ls = torch.log_softmax(
        (student_attn.to(f64) / float(temperature)).masked_fill(~key_mask, neg_inf), dim=-1)
    pt = lt.exp()
    kl = torch.where(pt > 0, pt * (lt - ls), torch.zeros((), dtype=f64)).sum(-1)
    attn_kl = (kl * q_mask).sum() / (q_mask.sum() * h)
    cls_mse = ((teacher_cls.to(f64) - student_cls.to(f64)) ** 2).mean()
    return {"attn_kl": attn_kl, "cls_mse": cls_mse, "loss": attn_kl + cls_mse}
