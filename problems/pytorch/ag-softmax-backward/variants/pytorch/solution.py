"""参考答案：沿 dim 做一次点积，再逐元素缩放，全程不出现 n×n 雅可比。

softmax 支路由 y_i(δ_ij − y_j) 收缩到 grad 上得 dx_i = y_i(grad_i − Σ_k y_k grad_k)；
log_softmax 支路因为 dx_i = grad_i − p_i Σ_k grad_k，只需 grad 的一次求和。两条路的
求和对象不同（一个用 p 加权、一个不用），写成同一个必然在其中一侧翻车。
"""

import torch


def _norm_dim(dim, ndim):
    d = dim + ndim if dim < 0 else dim
    if not 0 <= d < ndim:
        raise ValueError("dim 越界")
    return d


def solve(y, grad, dim=-1, log_input=False):
    if not isinstance(log_input, bool):
        raise ValueError("log_input 必须是 bool")
    if tuple(y.shape) != tuple(grad.shape):
        raise ValueError("y 与 grad 形状必须一致")
    d = _norm_dim(dim, y.dim())

    if log_input:
        p = y.double().exp()
        gd = grad.double()
        out = gd - p * gd.sum(dim=d, keepdim=True)
    else:
        yd = y.double()
        gd = grad.double()
        s = (gd * yd).sum(dim=d, keepdim=True)
        out = yd * (gd - s)
    return out.to(y.dtype)
