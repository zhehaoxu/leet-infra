"""参考答案：把四种求和格式按定义逐字实现，全程 IEEE754 double。

关键教学点是朴素 Kahan 与 Kahan–Babuska(Neumaier) 的差别：Kahan 的补偿项 c 是在
「s 已经比 v 大」的前提下推出来的，当新元素反而更大（例如先加 1e16 再加 -1e16）时
(t - s) 会把 v 的有效位彻底吃掉，c 变成 0，误差永久丢失。Neumaier 用 |s| 与 |v| 的
大小分支决定补哪一项，于是把这条漏洞堵上——示例 [1e16, 1, -1e16, 1] 正是分水岭：
naive 给 1.0、kahan 也给 1.0、kahan_babuska 给精确的 2.0。

另一个必须显式处理的坑：补偿项里的 (s - t) 在 s、t 都是 inf 时得到 nan，会把整个结果弄坏，
所以非有限值统一退化成 naive 顺序累加，让 IEEE 语义自己传播。
"""

import torch


def _flatten_float64(x):
    if not isinstance(x, torch.Tensor):
        raise ValueError("x 必须是 torch.Tensor")
    if x.dtype not in (torch.float32, torch.float64):
        raise ValueError("x 必须是 float32 或 float64 张量")
    return x.detach().to(torch.float64).reshape(-1).tolist()


def _naive(vals):
    s = 0.0
    for v in vals:
        s = s + v
    return s


def _kahan(vals):
    s = 0.0
    c = 0.0
    for v in vals:
        y = v - c
        t = s + y
        c = (t - s) - y
        s = t
    return s


def _kahan_babuska(vals):
    s = 0.0
    c = 0.0
    for v in vals:
        t = s + v
        if abs(s) >= abs(v):
            c += (s - t) + v
        else:
            c += (v - t) + s
        s = t
    return s + c


def _pairwise(vals, block):
    if block >= len(vals):
        return _naive(vals)
    parts = [_naive(vals[i:i + block]) for i in range(0, len(vals), block)]
    return _merge(parts)


def _merge(parts):
    if len(parts) == 1:
        return parts[0]
    mid = len(parts) // 2
    left = _merge(parts[:mid])
    right = _merge(parts[mid:])
    return left + right


def _all_finite(vals):
    for v in vals:
        if v != v or v in (float('inf'), float('-inf')):
            return False
    return True


def solve(x, mode='kahan_babuska', block=8):
    if not isinstance(mode, str):
        raise ValueError("mode 必须是字符串")
    if mode not in ('naive', 'kahan', 'kahan_babuska', 'pairwise'):
        raise ValueError(f"未知 mode: {mode}")
    b = int(block)
    if b < 1:
        raise ValueError("block 必须 >= 1")
    vals = _flatten_float64(x)
    if len(vals) == 0:
        return 0.0
    # 补偿项含 (s - t) 这类减法，遇到 inf 会算出 inf - inf = nan 污染结果，
    # 因此特殊值一律退化为 naive 的顺序累加。
    if not _all_finite(vals):
        return _naive(vals)
    if mode == 'naive':
        return _naive(vals)
    if mode == 'kahan':
        return _kahan(vals)
    if mode == 'kahan_babuska':
        return _kahan_babuska(vals)
    return _pairwise(vals, b)
