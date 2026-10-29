"""参考答案：受限 einsum 的「统一字母序对齐 → 广播逐元素乘 → 归约轴求和」实现。

关键手法：order = 被归约的字母（按首次出现顺序）+ 输出字母。每个操作数先 permute 到
它在 order 中拥有的字母顺序，再 reshape 成 len(order) 根轴（自己不拥有的轴长度取 1），
于是纯靠广播就把「外积 / 逐元素乘 / batch 对角」统一成一个乘积；求和轴恰好就是
order 的前 len(contracted) 根，剩下的轴天然已是输出顺序，无需再次 permute。
"""

import torch

ALLOWED = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _labels(part):
    return [c for c in part if c not in " ,"]


def solve(spec, *operands):
    if not isinstance(spec, str):
        raise ValueError("spec 必须是字符串")
    if "->" not in spec:
        raise ValueError("必须显式写出 '->' 输出标注")
    if "..." in spec:
        raise ValueError("不支持省略号 '...'")
    lhs, rhs = spec.split("->")
    in_parts = lhs.split(",")
    if len(in_parts) != len(operands):
        raise ValueError("标注段数与操作数个数不符")
    if not 1 <= len(operands) <= 3:
        raise ValueError("支持 1–3 个操作数")

    in_labels = []
    for part, t in zip(in_parts, operands):
        lst = _labels(part)
        if any(c not in ALLOWED for c in lst):
            raise ValueError("标注只允许单个字母")
        if len(lst) != len(set(lst)):
            raise ValueError("同一操作数内标注重复（不支持 trace/diagonal）")
        if len(lst) != t.dim():
            raise ValueError("标注字母数与张量维数不符")
        in_labels.append(lst)

    out_labels = _labels(rhs)
    if any(c not in ALLOWED for c in out_labels):
        raise ValueError("输出标注只允许单个字母")
    if len(out_labels) != len(set(out_labels)):
        raise ValueError("输出标注重复")
    seen = set()
    for lst in in_labels:
        seen |= set(lst)
    if not set(out_labels) <= seen:
        raise ValueError("输出标注含输入中不存在的字母")

    contracted = []
    for lst in in_labels:
        for c in lst:
            if c not in out_labels and c not in contracted:
                contracted.append(c)
    order = contracted + out_labels

    out_dtype = operands[0].dtype
    acc = None
    for lst, t in zip(in_labels, operands):
        sizes = dict(zip(lst, range(t.dim())))
        kept = [c for c in order if c in lst]
        perm = [lst.index(c) for c in kept]
        view_shape = [t.shape[sizes[c]] if c in sizes else 1 for c in order]
        view = t.double().permute(perm).reshape(view_shape)
        acc = view if acc is None else acc * view

    if contracted:
        acc = acc.sum(dim=tuple(range(len(contracted))))
    return acc.to(out_dtype)
