"""参考答案：torch.linalg.matrix_norm 等价实现（各分支都归到「与 x 同轴布局、被归约轴长度 1」，
最后按 keepdim 决定是否 squeeze）。

要点：
- "fro" 是 sqrt(Σa²)，与向量 2-范数同式，但别忘了开方；
- 诱导 1/-1 范数是「沿行轴求绝对值和 → 沿列轴取 max/min」，inf/-inf 恰好相反，
  这正是把矩阵范数与向量范数搞混的人最容易写反的一处；
- 2/-2/nuc 需要奇异值：把两根矩阵轴 movedim 到末尾后 svdvals，奇异值非负且降序，
  取 max/min/sum 都是确定数值，不存在奇异向量符号那种对拍歧义。
- movedim 后 batch 轴按原始升序排列，因此 svd 分支要把结果补回 x 的轴布局再统一 squeeze。
"""

import torch

_POS_INF = {"inf", float("inf")}
_NEG_INF = {"-inf", float("-inf")}


def _two_axes(x, dim):
    if not isinstance(dim, (tuple, list)) or len(dim) != 2:
        raise ValueError("dim 必须是恰好两个 int")
    axes = []
    for d in dim:
        if not isinstance(d, int):
            raise ValueError("dim 元素必须是 int")
        a = d if d >= 0 else d + x.dim()
        if not 0 <= a < x.dim():
            raise ValueError("dim 越界")
        axes.append(a)
    if axes[0] == axes[1]:
        raise ValueError("dim 两根轴不得相同")
    if x.dim() < 2:
        raise ValueError("matrix_norm 需要至少二维输入")
    return axes[0], axes[1]


def _restore(v, ndim, ra, ca):
    """v 的轴序 = 原张量中除 ra/ca 外升序保留的轴；补回两根长度 1 的轴放回原位。"""
    full = [1] * ndim
    kept = [i for i in range(ndim) if i not in (ra, ca)]
    for i, s in zip(kept, v.shape):
        full[i] = s
    return v.reshape(full)


def solve(x, ord="fro", dim=(-2, -1), keepdim=False):
    ra, ca = _two_axes(x, dim)
    xd = x.double()
    nd = x.dim()

    if ord == "fro":
        v = xd.pow(2).sum(dim=(ra, ca), keepdim=True).sqrt()
    elif ord in (1, -1) and not isinstance(ord, bool):
        col = xd.abs().sum(dim=ra, keepdim=True)          # 沿行轴求和：(..., 1, n)
        v = col.amax(dim=ca, keepdim=True) if ord == 1 else col.amin(dim=ca, keepdim=True)
    elif ord in _POS_INF or ord in _NEG_INF:
        row = xd.abs().sum(dim=ca, keepdim=True)          # 沿列轴求和：(..., m, 1)
        if ord in _POS_INF:
            v = row.amax(dim=ra, keepdim=True)
        else:
            v = row.amin(dim=ra, keepdim=True)
    elif ord in (2, -2, "nuc"):
        y = xd.movedim((ra, ca), (nd - 2, nd - 1))
        sv = torch.linalg.svdvals(y)                      # (..., min(m, n))，非负降序
        if ord == 2:
            vb = sv.max(dim=-1).values
        elif ord == -2:
            vb = sv.min(dim=-1).values
        else:
            vb = sv.sum(dim=-1)
        v = _restore(vb, nd, ra, ca)
    else:
        raise ValueError(f"不支持的 ord: {ord!r}")

    if not keepdim:
        v = v.squeeze(dim=(ra, ca))
    return v.to(x.dtype)
