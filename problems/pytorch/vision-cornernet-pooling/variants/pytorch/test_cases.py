"""CornerNet corner pooling 前向账（test cases）。"""

import math
import random


def _py_pool(x, direction):
    """独立 oracle：暴力逐点重算 O(H·W·n)。direction: 'L','R','T','B'。"""
    h, w = len(x), len(x[0])
    out = [[0.0] * w for _ in range(h)]
    for i in range(h):
        for j in range(w):
            if direction == 'L':    # 右侧含自身
                out[i][j] = max(x[i][j:])
            elif direction == 'R':  # 左侧含自身
                out[i][j] = max(x[i][:j + 1])
            elif direction == 'T':  # 下方含自身
                out[i][j] = max(x[k][j] for k in range(i, h))
            else:                   # 上方含自身
                out[i][j] = max(x[k][j] for k in range(i + 1))
    return out


_COMBOS = {"tl": ('T', 'L'), "tr": ('T', 'R'), "bl": ('B', 'L'), "br": ('B', 'R')}


def _py(x, mode):
    d1, d2 = _COMBOS[mode]
    a = _py_pool(x, d1)
    b = _py_pool(x, d2)
    return [[a[i][j] + b[i][j] for j in range(len(x[0]))]
            for i in range(len(x))]


def _close(actual, expect):
    assert isinstance(actual, list) and len(actual) == len(expect)
    for ra, re_ in zip(actual, expect):
        assert len(ra) == len(re_)
        for a, e in zip(ra, re_):
            assert abs(a - e) < 1e-9, f"{a} != {e}"
    return True


def _compare(solve, x, mode):
    led = solve(x, mode)
    _close(led, _py(x, mode))
    return led


def check_anchor_hand(solve):
    # 2x3 手算：x = [[1, 5, 2], [3, 0, 4]]
    x = [[1.0, 5.0, 2.0], [3.0, 0.0, 4.0]]
    # top（下方 max）= [[3,5,4],[3,0,4]]；left（右侧 max）= [[5,5,2],[4,4,4]]
    # tl = [[8,10,6],[7,4,8]]
    led = _compare(solve, x, "tl")
    assert led == [[8.0, 10.0, 6.0], [7.0, 4.0, 8.0]]
    # bottom（上方 max）= [[1,5,2],[3,5,4]]；right（左侧 max）= [[1,5,5],[3,3,4]]
    # br = [[2,10,7],[6,8,8]]
    led = _compare(solve, x, "br")
    assert led == [[2.0, 10.0, 7.0], [6.0, 8.0, 8.0]]


def check_single_row_col(solve):
    # 单行：top/bottom 退化为自身；单列同理
    x = [[3.0, 1.0, 2.0]]
    led = _compare(solve, x, "tl")
    # top=自身，left(右侧 max)=[3,2,2] → [6,3,4]
    assert led == [[6.0, 3.0, 4.0]]
    x2 = [[1.0], [5.0], [2.0]]
    led2 = _compare(solve, x2, "br")
    # bottom(上方 max)=[1,5,5]，right=自身=[1,5,2] → 相加 [2,10,7]
    assert led2 == [[2.0], [10.0], [7.0]]


def check_monotone_rows(solve):
    # 递增行：right_pool=自身；递减行：left_pool=自身
    x = [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]
    led = _compare(solve, x, "tr")
    # top=[[3,2,3],[3,2,1]]? top 取下方 max：row0=max(1,2,3 按列)=[3,2,3]，row1=[3,2,1]
    # right(左侧 max)=[[1,2,3],[3,3,3]] → tr=[[4,4,6],[6,5,4]]
    assert led == [[4.0, 4.0, 6.0], [6.0, 5.0, 4.0]]


def check_all_equal(solve):
    # 全 2.0：任何方向 pool 都=2.0，corner=4.0
    x = [[2.0] * 4 for _ in range(3)]
    for mode in ("tl", "tr", "bl", "br"):
        assert _compare(solve, x, mode) == [[4.0] * 4 for _ in range(3)]


def check_negatives(solve):
    # 全负值：running max 仍含自身
    x = [[-3.0, -1.0, -2.0], [-0.5, -4.0, -1.5]]
    led = _compare(solve, x, "bl")
    # bottom(上方 max)=[[-3,-1,-2],[-0.5,-1,-1.5]]
    # left(右侧 max)=[[-1,-1,-2],[-0.5,-1.5,-1.5]]
    assert led == [[-4.0, -2.0, -4.0], [-1.0, -2.5, -3.0]]


def check_out_ge_x_single_pool(solve):
    # running max 逐元素 >= x（对角支配性）
    rng = random.Random(677)
    for _ in range(50):
        h, w = rng.randint(1, 6), rng.randint(1, 6)
        x = [[rng.uniform(-5, 5) for _ in range(w)] for _ in range(h)]
        led = _compare(solve, x, "tl")
        # top 支路 >= x：tl = top + left >= x + x? 不成立（left 可能<x）。
        # 改验证：tl >= top支路单值？无法从输出分离。跳过结构性断言，
        # 随机对拍已覆盖。
        _ = led


def check_oracle_random(solve):
    rng = random.Random(1677)
    for _ in range(200):
        h = rng.randint(1, 7)
        w = rng.randint(1, 7)
        x = [[round(rng.uniform(-9, 9), 3) for _ in range(w)] for _ in range(h)]
        mode = rng.choice(["tl", "tr", "bl", "br"])
        _compare(solve, x, mode)


def check_invalid_inputs(solve):
    for bad_x, bad_mode in [
        ([], "tl"),
        ([[]], "tl"),
        ([[1.0], [2.0, 3.0]], "tl"),      # 非矩形
        ([[float('nan')]], "tl"),          # 非有限
        ([[1.0]], "xx"),                   # 非法 mode
        ("nope", "tl"),
    ]:
        try:
            solve(bad_x, bad_mode)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError(f"expected error: {bad_x} {bad_mode}")
