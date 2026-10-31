"""CornerNet corner pooling 前向账（solution）。

转写官方 _cpools/src/{left,right,top,bottom}_pool.cpp forward 循环。
"""

import math


def _validate(x):
    if not isinstance(x, list) or not x or not isinstance(x[0], list) \
            or not x[0]:
        raise ValueError("x must be a non-empty 2-D list")
    w = len(x[0])
    for row in x:
        if not isinstance(row, list) or len(row) != w:
            raise ValueError("x must be rectangular")
        for v in row:
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError("x values must be finite floats")


def right_pool(x):
    """out[i][j] = max(x[i][0..j])：向右扫描累积（官方 right_pool.cpp）。"""
    h, w = len(x), len(x[0])
    out = [[0.0] * w for _ in range(h)]
    for i in range(h):
        out[i][0] = x[i][0]
        for j in range(1, w):
            out[i][j] = x[i][j] if x[i][j] > out[i][j - 1] else out[i][j - 1]
    return out


def left_pool(x):
    """out[i][j] = max(x[i][j..])：向左扫描累积（官方 left_pool.cpp）。"""
    h, w = len(x), len(x[0])
    out = [[0.0] * w for _ in range(h)]
    for i in range(h):
        out[i][w - 1] = x[i][w - 1]
        for j in range(w - 2, -1, -1):
            out[i][j] = x[i][j] if x[i][j] > out[i][j + 1] else out[i][j + 1]
    return out


def top_pool(x):
    """out[i][j] = max(x[i..][j])：向下扫描累积（官方 top_pool.cpp）。"""
    h, w = len(x), len(x[0])
    out = [[0.0] * w for _ in range(h)]
    for j in range(w):
        out[h - 1][j] = x[h - 1][j]
        for i in range(h - 2, -1, -1):
            out[i][j] = x[i][j] if x[i][j] > out[i + 1][j] else out[i + 1][j]
    return out


def bottom_pool(x):
    """out[i][j] = max(x[0..i][j])：向上扫描累积（官方 bottom_pool.cpp）。"""
    h, w = len(x), len(x[0])
    out = [[0.0] * w for _ in range(h)]
    for j in range(w):
        out[0][j] = x[0][j]
        for i in range(1, h):
            out[i][j] = x[i][j] if x[i][j] > out[i - 1][j] else out[i - 1][j]
    return out


_POOLS = {"tl": (top_pool, left_pool), "tr": (top_pool, right_pool),
          "bl": (bottom_pool, left_pool), "br": (bottom_pool, right_pool)}


def solve(x, mode):
    _validate(x)
    if mode not in _POOLS:
        raise ValueError(f"mode must be one of {sorted(_POOLS)}")
    p1, p2 = _POOLS[mode]
    a, b = p1(x), p2(x)
    return [[a[i][j] + b[i][j] for j in range(len(x[0]))]
            for i in range(len(x))]
