"""参考答案：硬掩码 STE 与 WSN（clamp(|w|/τ,0,1)）软掩码的前向/反向。"""

import torch


def solve(weight, tau, grad_out, mode):
    abs_w = weight.abs()
    if mode == "hard_ste":
        mask = (abs_w >= tau).to(weight.dtype)
        grad = grad_out * mask
    elif mode == "wsn":
        ratio = abs_w / tau
        mask = torch.clamp(ratio, 0.0, 1.0)
        grad = grad_out * (mask + ratio * (abs_w < tau).to(weight.dtype))
    else:
        raise ValueError(f"unknown mode: {mode}")
    return weight * mask, grad, mask
