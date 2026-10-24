import torch


def solve(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm：对 x 的最后一维做均方根归一化，再逐元素乘 weight。

    参数
    ----
    x      : (..., d) 待归一化张量
    weight : (d,)     逐元素可学习权重（HF 的 RMSNorm.weight）
    eps    : 加在均方根内的极小值

    返回与 x 同形状、同 dtype 的归一化结果；须在 float32 下算方差/归一化再转回原 dtype。
    """
    # TODO: 实现 RMSNorm（不减均值，只用 1/d * sum(x^2) 做缩放；eps 在根号内）
    raise NotImplementedError
