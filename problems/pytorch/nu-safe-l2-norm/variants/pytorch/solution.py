"""安全 p 范数：先缩放、再归约（BLAS dnrm2 / lapack lacn2 的那条老路子）。

关键点是「归约之前把每一项压进 [0, 1]」：
    m = max|x|,  ||x||_p = m * (Σ (|x|/m)^p)^(1/p)
不缩放直接求幂和的话，3e154 平方成 9e308 已经贴到 float64 天花板、4e154 直接上溢，
而 3e-200 平方成 9e-400 整个抹成 0——torch 自带的 norm 就同时踩了这两个坑。

实现细节上有一处容易写坏：缩放因子 m 为 0 或 inf 时不能真去除（0/0 与 inf/inf 都是 NaN），
所以先算出「这段能不能走缩放公式」的掩码，除以一个占位的 1.0，最后再用掩码把特殊段填回去。
"""
import math

import torch

INF = float('inf')
NAN = float('nan')


def _check(x, p, dim, keepdim):
    if not isinstance(x, torch.Tensor):
        raise ValueError('x must be a Tensor')
    if x.dtype not in (torch.float32, torch.float64):
        raise ValueError('x must be float32 or float64')
    if isinstance(p, bool) or not isinstance(p, (int, float)):
        raise ValueError('p must be an int or float')
    if (isinstance(p, float) and math.isnan(p)) or p <= 0:
        raise ValueError('p must be positive (math.inf is allowed)')
    if dim is not None and (isinstance(dim, bool) or not isinstance(dim, int)):
        raise ValueError('dim must be None or an int')
    if not isinstance(keepdim, bool):
        raise ValueError('keepdim must be a bool')


def solve(x, p=2.0, dim=None, keepdim=False):
    _check(x, p, dim, keepdim)
    out_dtype = x.dtype
    work = x.detach().to(torch.float64)
    a = work.abs()
    if dim is None:
        axes = (0,)
        a = a.reshape(-1)
        keep_shape = ()
    else:
        ax = dim + a.dim() if dim < 0 else dim
        if not 0 <= ax < a.dim():
            raise ValueError('dim %d is out of range for a %d-D tensor' % (dim, a.dim()))
        axes = (ax,)
        if keepdim:
            keep_shape = tuple(1 if i == ax else s for i, s in enumerate(a.shape))
        else:
            keep_shape = tuple(s for i, s in enumerate(a.shape) if i != ax)

    if a.shape[axes[0]] == 0:            # 空段：长度 0 的段既没得 max 也没得加
        return torch.zeros(keep_shape, dtype=out_dtype)

    m = a.amax(dim=axes, keepdim=True)
    has_nan = a.isnan().sum(dim=axes, keepdim=True) > 0
    scalable = torch.isfinite(m) & (m > 0)      # m == 0 / inf / nan 都走特殊填充分支
    safe_m = torch.where(scalable, m, torch.ones_like(m))
    if p == INF:
        scaled = safe_m                           # ||x||_inf 就是缩放因子本身
    else:
        scaled = safe_m * (a / safe_m).pow(p).sum(dim=axes, keepdim=True).pow(1.0 / p)
    fill = torch.where(has_nan, torch.full_like(m, NAN),
                       torch.where(m == INF, torch.ones_like(m) * INF, torch.zeros_like(m)))
    out = torch.where(scalable, scaled, fill)
    return out.reshape(keep_shape).to(out_dtype)
