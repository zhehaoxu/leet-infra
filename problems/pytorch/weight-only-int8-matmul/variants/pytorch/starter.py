import torch


def solve(x: torch.Tensor, Wq: torch.Tensor, w_scale: torch.Tensor) -> torch.Tensor:
    """weight-only int8：把 per-channel 反量化的 Linear 用整型权重算出来。

    x [M,K] 浮点激活（不量化）、Wq [N,K] int8 权重、w_scale [N] 每输出通道 scale。
    out[m,n] = w_scale[n] * sum_k x[m,k]*Wq[n,k]。返回 [M,N]，dtype 同 x。
    等价于 F.linear(x, Wq.float()*w_scale[:,None])。
    """
    # TODO: 实现 weight-only int8 反量化矩阵乘
    raise NotImplementedError
