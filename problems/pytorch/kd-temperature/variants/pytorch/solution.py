"""参考答案：log_softmax(z/T)，逐样本 KL(teacher||student) 乘 T^2，按 reduction 归约。"""

import torch


def solve(student_logits, teacher_logits, T, reduction="mean"):
    ts = torch.log_softmax(student_logits / T, dim=-1)
    tt = torch.log_softmax(teacher_logits / T, dim=-1)
    p_t = tt.exp()
    kl_rows = (p_t * (tt - ts)).sum(dim=-1)
    loss = (T * T) * kl_rows
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    raise ValueError(f"unknown reduction: {reduction!r}")
