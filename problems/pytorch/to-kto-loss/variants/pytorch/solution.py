"""参考答案：KTO 前景理论损失。"""

import torch


def solve(log_ratios, labels, z_ref, beta, lambda_D, lambda_U):
    desirable = labels > 0.5
    v_good = torch.sigmoid(beta * (log_ratios - z_ref))
    v_bad = torch.sigmoid(beta * (z_ref - log_ratios))
    value = torch.where(desirable, v_good, v_bad)
    lam = torch.where(desirable, torch.as_tensor(lambda_D), torch.as_tensor(lambda_U))
    return (lam * (1 - value)).mean()
