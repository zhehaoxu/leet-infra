"""参考答案：torch.diff 等价实现（先按 prepend/x/append 拼接，再做 n 次相邻相减）。

n 阶差分用 narrow 做「去掉首元素」减「去掉尾元素」，比切片 [:, 1:] − [:, :-1] 更通用
（切片写法只能处理 dim=-1，narrow 对任意轴都成立）。每次做完轴长减 1，故总长度
= 拼接后长度 − n，长度非正时直接报错而不是交给判题一个空张量。
"""

import torch


def _check_side(name, t, x, axis):
    if not isinstance(t, torch.Tensor):
        raise ValueError(f"{name} 必须是张量")
    if t.dim() != x.dim():
        raise ValueError(f"{name} 与 x 的维数不同")
    for i in range(x.dim()):
        if i != axis and t.shape[i] != x.shape[i]:
            raise ValueError(f"{name} 第 {i} 维与 x 不符")
    if t.dtype != x.dtype:
        raise ValueError(f"{name} 的 dtype 必须与 x 相同")


def solve(x, n=1, dim=-1, prepend=None, append=None):
    if isinstance(n, bool) or not isinstance(n, int):
        raise ValueError("n 必须是 int")
    if n < 0:
        raise ValueError("n 必须非负")
    if x.dim() == 0:
        raise ValueError("diff 需要至少一维输入")
    axis = dim if dim >= 0 else dim + x.dim()
    if not 0 <= axis < x.dim():
        raise ValueError("dim 越界")

    if prepend is not None:
        _check_side("prepend", prepend, x, axis)
    if append is not None:
        _check_side("append", append, x, axis)

    if n == 0:
        return x.clone()

    parts = []
    if prepend is not None:
        parts.append(prepend)
    parts.append(x)
    if append is not None:
        parts.append(append)
    y = parts[0] if len(parts) == 1 else torch.cat(parts, dim=axis)

    for _ in range(n):
        size = y.shape[axis]
        if size < 2:
            raise ValueError("差分后沿 dim 轴的长度非正")
        y = y.narrow(axis, 1, size - 1) - y.narrow(axis, 0, size - 1)
    return y
