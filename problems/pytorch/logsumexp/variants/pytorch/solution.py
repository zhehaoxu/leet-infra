"""参考答案：torch.logsumexp 等价实现（提最大值 + ±inf/NaN 显式分支）。

m = amax 之后只有 m 有限才走 m + log Σexp(x−m)；m = +inf（段内有 +inf）与
m = −inf（整段全 −inf）都必须直接取 m 的值，否则 x−m 产生 NaN 污染整段。
NaN 也落在这条「非有限」分支上，天然完成 NaN 传播。
"""

import torch


def _axes(x, dim):
    if dim is None:
        return tuple(range(x.dim()))
    if isinstance(dim, int) and not isinstance(dim, bool):
        dims = (dim,)
    elif isinstance(dim, (tuple, list)):
        dims = tuple(dim)
    else:
        raise ValueError("dim 必须是 None、int 或 int 元组")
    if len(dims) == 0:
        raise ValueError("dim 不能为空")
    axes = []
    for d in dims:
        if not isinstance(d, int) or isinstance(d, bool):
            raise ValueError("dim 元素必须是 int")
        a = d if d >= 0 else d + x.dim()
        if not 0 <= a < x.dim():
            raise ValueError("dim 越界")
        if a in axes:
            raise ValueError("dim 含重复轴")
        axes.append(a)
    return tuple(sorted(axes))


def solve(x, dim=None, keepdim=False):
    axes = _axes(x, dim)
    xd = x.double()
    m = xd.amax(dim=axes, keepdim=True)
    finite_m = torch.isfinite(m)
    safe_m = torch.where(finite_m, m, torch.zeros_like(m))
    s = (xd - safe_m).exp().sum(dim=axes, keepdim=True)
    out = torch.where(finite_m, safe_m + s.log(), m)

    if not keepdim:
        out = out.squeeze(dim=axes)
    return out.to(x.dtype)
