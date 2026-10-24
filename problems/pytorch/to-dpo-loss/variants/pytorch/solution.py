import torch
import torch.nn.functional as F


def solve(chosen_logp, chosen_ref_logp, rejected_logp, rejected_ref_logp, beta):
    """DPO 直接偏好优化损失。"""
    logits = beta * ((chosen_logp - chosen_ref_logp) - (rejected_logp - rejected_ref_logp))
    return -F.logsigmoid(logits).mean()
