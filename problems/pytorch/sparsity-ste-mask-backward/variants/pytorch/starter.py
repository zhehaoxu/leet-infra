import torch


def solve(weight, tau, grad_out, mode):
    """剪枝-微调的单步前向 + 反向（上游梯度 grad_out 显式给定，无需 autograd）。
    mode="hard_ste"：m = 1[|w| >= tau]；y = w·m；dL/dw = g·m（掩码当常数，直通估计 STE）。
    mode="wsn"：    m = clamp(|w|/tau, 0, 1)；y = w·m；dL/dw = g·(m + |w|/tau·1[|w| < tau])。
    weight/grad_out float32 同形状；tau > 0 标量。返回 (y, grad_w, mask)，三者同形状 float32。
    禁止 torch.autograd。"""
    # TODO
    raise NotImplementedError
