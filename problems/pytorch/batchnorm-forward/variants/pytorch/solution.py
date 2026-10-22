"""参考答案：BatchNorm 前向 —— 训练用 batch 统计并更新 running stats，推理用 running stats。"""

import torch


def solve(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
):
    # 归约维：除通道维(dim=1)外的所有维（2D 输入即 (N,)）
    stats_dims = (0, *range(2, x.dim()))
    if training:
        mu = x.mean(stats_dims)
        var = x.var(stats_dims, unbiased=False)  # 归一化用有偏方差
    else:
        mu, var = running_mean, running_var
    shape = [1] * x.dim()
    shape[1] = -1
    y = (x - mu.view(shape)) / torch.sqrt(var.view(shape) + eps)
    y = y * weight.view(shape) + bias.view(shape)
    if training:
        # 陷阱：PyTorch 的 running_var 存的是无偏方差（乘 N/(N-1)）
        n = x.numel() // x.shape[1]
        var_running = var * n / (n - 1) if n > 1 else var
        new_rm = (1.0 - momentum) * running_mean + momentum * mu
        new_rv = (1.0 - momentum) * running_var + momentum * var_running
    else:
        new_rm, new_rv = running_mean, running_var
    return y, new_rm, new_rv
