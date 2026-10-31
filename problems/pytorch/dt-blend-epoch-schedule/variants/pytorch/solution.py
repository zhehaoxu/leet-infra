"""参考解：温度缩放 → 归一化 → 最大余额配额 → 平滑排布 → 循环越采样 → 全局编号。

四个容易写歪的口径点：

1. **篇数为 0 的源在缩放之前就被置零**，不是归一化之后再剔除；它的权重再大也进不了
   `total`。全部源都拿不到非零条目时 `total == 0`，无法归一化，直接非法。
2. **浮点结合顺序是契约的一部分**：`total` 按下标升序逐项左加，`probs[i] = e[i] / total`
   是除法而不是乘倒数，`q[i] = probs[i] * num_samples`。最大余额的平局规则写成
   「余额降序、下标升序」，用一次稳定排序即可，不需要额外循环。
3. **排布是「按源下标升序逐源放置」**，理想槽用整数除法 `j * num_samples // counts[i]`
   （与 `floor(j * num_samples / counts[i])` 恒等，且不会碰到浮点舍入），再从理想槽
   向后线性扫描第一个空闲槽。换一种放置顺序（按配额大小、按概率）得到的序列不同。
4. **文档号是发射计数对篇数取模**，第 `j` 次发射读 `j % sizes[i]`，配额超过篇数就循环
   复读；`flat_indices` 只是把源内编号搬到 exclusive 前缀和上。

校验里两处刻意严格：`bool` 既不当 `int` 也不当 `float`；权重必须先能转成 `float`
（`float(10**400)` 会 `OverflowError`），且 `w ** expo` 不能溢出——负权重会被直接拒掉，
否则 `(-2.0) ** 0.5` 在 Python 里返回**复数**而不是报错。
"""

import math


def _as_float(value, why):
    try:
        wide = float(value)
    except (OverflowError, ValueError, TypeError):
        raise ValueError(why)
    if not math.isfinite(wide):
        raise ValueError(why)
    return wide


def _check(sizes, weights, num_samples, temperature):
    if not isinstance(sizes, list) or not sizes:
        raise ValueError("sizes must be a non-empty list of ints")
    for value in sizes:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("sizes entries must be non-negative ints")
    if not isinstance(weights, list) or len(weights) != len(sizes):
        raise ValueError("weights must be a list as long as sizes")
    for value in weights:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("weights entries must be ints or floats")
        if value < 0:
            raise ValueError("weights must be non-negative")
        _as_float(value, "weights must be finite floats")
    if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples < 0:
        raise ValueError("num_samples must be a non-negative int")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("temperature must be an int or float")
    temp = _as_float(temperature, "temperature must be finite and positive")
    if temp <= 0.0:
        raise ValueError("temperature must be positive")
    return temp


def _entries(sizes, weights, expo):
    entries = []
    for index, count in enumerate(sizes):
        if count == 0:
            entries.append(0.0)
            continue
        try:
            scaled = float(weights[index]) ** expo
        except OverflowError:
            raise ValueError("temperature scaling overflows")
        entries.append(scaled)
    return entries


def _normalise(entries):
    total = 0.0
    for value in entries:
        total += value
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("the temperature weights must sum to a positive finite total")
    return [value / total for value in entries]


def _quota(probs, num_samples):
    ideal = [probs[i] * num_samples for i in range(len(probs))]
    counts = [math.floor(value) for value in ideal]
    extra = num_samples - sum(counts)
    ranking = sorted(range(len(probs)), key=lambda i: (-(ideal[i] - counts[i]), i))
    for index in ranking[:extra]:
        counts[index] += 1
    return counts


def _place(counts, num_samples, sizes):
    schedule = [-1] * num_samples
    doc_indices = [-1] * num_samples
    for source, quota in enumerate(counts):
        for emission in range(quota):
            slot = emission * num_samples // quota
            while schedule[slot] != -1:
                slot += 1
            schedule[slot] = source
            doc_indices[slot] = emission % sizes[source]
    return schedule, doc_indices


def _offsets(sizes):
    offsets = []
    running = 0
    for count in sizes:
        offsets.append(running)
        running += count
    return offsets


def solve(sizes, weights, num_samples, temperature):
    temp = _check(sizes, weights, num_samples, temperature)
    expo = 1.0 / temp
    probs = _normalise(_entries(sizes, weights, expo))
    counts = _quota(probs, num_samples)
    schedule, doc_indices = _place(counts, num_samples, sizes)
    offsets = _offsets(sizes)
    flat_indices = [offsets[schedule[t]] + doc_indices[t] for t in range(num_samples)]
    return probs, counts, schedule, doc_indices, flat_indices
