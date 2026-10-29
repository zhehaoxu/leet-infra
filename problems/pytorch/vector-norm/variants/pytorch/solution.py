"""参考答案：torch.linalg.vector_norm 等价实现（float64 累计 + 提最大值的稳定 p-范数）。

三处易错点：
1. ord=0 是「非零个数」而非 (Σ|x|^0)^(1/0)，不能套幂次公式；
2. 只有 ord=1 是 Σ|x|；ord=-1 按 p=−1 的幂次公式 (Σ|x|^{−1})^{−1}，与向量的 1-范数不同；
3. 有限 p 直接 Σ|x|^p 会在 |x|~1e3、p=8 时上溢，故提出 m=max|x| 后算
   m·(Σ(|x|/m)^p)^(1/p)，该恒等式对负 p 同样成立（(m^p)^{1/p}=m, m>0）。
   负 p 时段内若有 0 元素，(|x|/m)^p = +inf → inf^{1/p} = 0 → 结果 0，与 torch 一致。
"""

import torch

_POS_INF = {"inf", float("inf")}
_NEG_INF = {"-inf", float("-inf")}


def _norm_axes(x, dim):
    if dim is None:
        return tuple(range(x.dim()))
    if isinstance(dim, int):
        dims = (dim,)
    elif isinstance(dim, (tuple, list)):
        dims = tuple(dim)
    else:
        raise ValueError("dim 必须是 int 或 int 元组")
    if len(dims) == 0:
        raise ValueError("dim 不能为空元组")
    axes = []
    for d in dims:
        if not isinstance(d, int):
            raise ValueError("dim 元素必须是 int")
        a = d if d >= 0 else d + x.dim()
        if not 0 <= a < x.dim():
            raise ValueError("dim 越界")
        if a in axes:
            raise ValueError("dim 含重复轴")
        axes.append(a)
    return tuple(sorted(axes))


def solve(x, ord=None, dim=None, keepdim=False):
    axes = _norm_axes(x, dim)
    absx = x.double().abs()

    if ord is None or ord == 2:
        out = absx.pow(2).sum(dim=axes, keepdim=True).sqrt()
    elif isinstance(ord, str) and ord in ("fro", "nuc"):
        raise ValueError(f"向量范数不支持 ord={ord!r}")
    elif ord in _POS_INF:
        out = absx.amax(dim=axes, keepdim=True)
    elif ord in _NEG_INF:
        out = absx.amin(dim=axes, keepdim=True)
    elif ord == 1:
        out = absx.sum(dim=axes, keepdim=True)
    elif ord == 0:
        out = (absx != 0).to(torch.float64).sum(dim=axes, keepdim=True)
    else:
        p = float(ord)
        if p != p or p in _POS_INF or p in _NEG_INF:
            raise ValueError("ord 必须是有限实数、0、1、2 或 inf/-inf")
        m = absx.amax(dim=axes, keepdim=True)
        scaled = torch.where(m > 0, absx / torch.where(m > 0, m, torch.ones_like(m)),
                             torch.zeros_like(absx))
        out = m * scaled.pow(p).sum(dim=axes, keepdim=True).pow(1.0 / p)

    if not keepdim:
        out = out.squeeze(dim=tuple(a for a in axes))
    return out.to(x.dtype)
