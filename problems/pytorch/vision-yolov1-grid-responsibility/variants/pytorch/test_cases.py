"""YOLOv1 网格责任编码账（test cases）。

独立 oracle：数学等价的直接式实现（decode 直接用截断后的 x/y/w/h，
不走 encode↔decode 往返），与 solution 的 C 链条式重放互证。
"""

import math
import random


def _py_solve(side, classes, labels, jitter, flip):
    dx, dy, sx, sy = jitter
    occ = {}
    dec = {}
    dropped = []
    for i, lab in enumerate(labels):
        lid, cx, cy, cw, ch = lab
        l = max(0.0, min(1.0, (cx - cw / 2.0) * sx - dx))
        r = max(0.0, min(1.0, (cx + cw / 2.0) * sx - dx))
        t = max(0.0, min(1.0, (cy - ch / 2.0) * sy - dy))
        b = max(0.0, min(1.0, (cy + ch / 2.0) * sy - dy))
        if flip:
            l, r = 1.0 - r, 1.0 - l
        x, y = (l + r) / 2.0, (t + b) / 2.0
        w, h = r - l, b - t
        if not (0.0 < x < 1.0 and 0.0 < y < 1.0):
            dropped.append({"index": i, "reason": "center_out"})
            continue
        gx, gy = int(x * side), int(y * side)
        tx, ty = x * side - gx, y * side - gy
        if w < 0.01 or h < 0.01:
            dropped.append({"index": i, "reason": "too_small"})
            continue
        if (gy, gx) in occ:
            dropped.append({"index": i, "reason": "occupied"})
            continue
        occ[(gy, gx)] = {"id": lid, "x": tx, "y": ty,
                         "w": math.sqrt(w), "h": math.sqrt(h), "conf": 1.0}
        dec[(gy, gx)] = [x, y, w, h]     # 数学等价：decode 应还原回这组值
    return {
        "grid": {f"{r},{c}": v for (r, c), v in occ.items()},
        "decoded": {f"{r},{c}": v for (r, c), v in dec.items()},
        "dropped": dropped,
    }


def _close(actual, expect, tol=1e-9):
    if isinstance(expect, dict):
        assert isinstance(actual, dict), f"not dict: {actual!r}"
        assert len(actual) == len(expect), f"key mismatch: {sorted(actual)} vs {sorted(expect)}"
        for k in expect:
            assert _close(actual[k], expect[k], tol), f"mismatch at key {k}"
        return True
    if isinstance(expect, list):
        assert isinstance(actual, list) and len(actual) == len(expect), \
            f"{actual!r} vs {expect!r}"
        return all(_close(a, e, tol) for a, e in zip(actual, expect))
    if isinstance(expect, float):
        assert abs(actual - expect) < tol, f"{actual} != {expect}"
        return True
    assert actual == expect, f"{actual!r} != {expect!r}"
    return True


def _compare(solve, side, classes, labels, jitter=None, flip=False):
    led = solve(side, classes, labels, jitter, flip)
    ref = _py_solve(side, classes, labels, jitter or [0.0, 0.0, 1.0, 1.0], flip)
    _close(led, ref)
    return led


def check_basic_encoding(solve):
    # S=7 手算：中心 (0.55,0.6) → col=3, row=4, tx≈0.85, ty≈0.2；sqrt 编码
    led = _compare(solve, 7, 20, [[3, 0.55, 0.6, 0.2, 0.4]])
    assert list(led["grid"]) == ["4,3"]
    cell = led["grid"]["4,3"]
    assert cell["id"] == 3
    _close(cell["x"], 0.85)
    _close(cell["y"], 0.2)
    _close(cell["w"], math.sqrt(0.2))
    _close(cell["h"], math.sqrt(0.4))
    assert cell["conf"] == 1.0
    # decode 往返：平移 + 平方
    _close(led["decoded"]["4,3"], [0.55, 0.6, 0.2, 0.4])


def check_jitter_and_constrain_recompute(solve):
    # 纯 jitter 平移：dx=0.1 使中心 0.5→0.4，格随动
    led = _compare(solve, 7, 20, [[0, 0.5, 0.5, 0.2, 0.2]], [0.1, 0.0, 1.0, 1.0])
    cell = led["grid"]["3,2"]       # row=3, col=2
    _close(cell["x"], 0.4 * 7 - 2)
    # 截断重算账点：left=x−w/2=−0.1 被 constrain 到 0 → w=right−0=0.2，
    # 中心变为 0.1（不是对原 w 的 clamp）
    led = _compare(solve, 7, 20, [[1, 0.05, 0.5, 0.3, 0.2]])
    cell = led["grid"]["3,0"]
    _close(cell["x"], 0.7)          # tx = 0.1*7 − 0
    _close(cell["w"], math.sqrt(0.2))
    _close(led["decoded"]["3,0"][0], 0.1)
    _close(led["decoded"]["3,0"][2], 0.2)


def check_flip(solve):
    # flip：left,right = 1−right, 1−left → 中心 0.3→0.7，w 不变
    led = _compare(solve, 7, 20, [[0, 0.3, 0.5, 0.2, 0.1]], flip=True)
    cell = led["grid"]["3,4"]
    _close(cell["x"], 0.7 * 7 - 4)
    _close(cell["w"], math.sqrt(0.2))


def check_center_boundary_drop(solve):
    # jitter 把中心精确推到 0 / 1：半开门丢弃（不是 clamp）
    led = _compare(solve, 7, 20, [[0, 0.1, 0.5, 0.2, 0.2]], [0.2, 0.0, 1.0, 1.0])
    assert led["dropped"] == [{"index": 0, "reason": "center_out"}]
    assert led["grid"] == {}
    led = _compare(solve, 7, 20, [[0, 0.1, 0.5, 0.2, 0.2]], [1.0, 0.0, 1.0, 1.0])
    assert led["dropped"] == [{"index": 0, "reason": "center_out"}]
    # dx=−0.8：left=0.8, right=1.0 → 中心 0.9 不丢
    led = _compare(solve, 7, 20, [[0, 0.1, 0.5, 0.2, 0.2]], [-0.8, 0.0, 1.0, 1.0])
    assert led["dropped"] == []


def check_grid_line_center(solve):
    # 中心恰在格线：x=0.5, S=2 → col=1, tx=0.0（合法落格，不丢）
    led = _compare(solve, 2, 5, [[2, 0.5, 0.75, 0.1, 0.1]])
    assert list(led["grid"]) == ["1,1"]
    cell = led["grid"]["1,1"]
    assert cell["x"] == 0.0
    _close(cell["y"], 0.5)
    _close(led["decoded"]["1,1"], [0.5, 0.75, 0.1, 0.1])


def check_small_box_drop(solve):
    # 宽高门严格 < 0.01：w=0.01 保留，w=0.009 丢弃
    led = _compare(solve, 7, 20, [[0, 0.5, 0.5, 0.01, 0.3]])
    assert led["dropped"] == []
    _close(led["grid"]["3,3"]["w"], 0.1)
    led = _compare(solve, 7, 20, [[0, 0.5, 0.5, 0.009, 0.3]])
    assert led["dropped"] == [{"index": 0, "reason": "too_small"}]
    led = _compare(solve, 7, 20, [[0, 0.5, 0.5, 0.3, 0.009]])
    assert led["dropped"] == [{"index": 0, "reason": "too_small"}]
    # 浮点精确 w==0.01（l=0.0, r=0.01）：边界保留（< 严格）
    led = _compare(solve, 7, 20, [[0, 0.005, 0.5, 0.01, 0.3]])
    assert led["dropped"] == []
    _close(led["grid"]["3,0"]["w"], 0.1)


def check_occupied_first_wins(solve):
    # S=1 全图一格：第二条同格 GT 被先到先得仲裁丢弃
    labels = [[0, 0.3, 0.3, 0.2, 0.2], [4, 0.7, 0.7, 0.2, 0.2]]
    led = _compare(solve, 1, 20, labels)
    assert list(led["grid"]) == ["0,0"]
    assert led["grid"]["0,0"]["id"] == 0
    assert led["dropped"] == [{"index": 1, "reason": "occupied"}]


def check_empty_labels(solve):
    led = _compare(solve, 7, 20, [])
    assert led == {"grid": {}, "decoded": {}, "dropped": []}


def check_oracle_random(solve):
    rng = random.Random(701)
    for _ in range(300):
        side = rng.choice([1, 2, 3, 5, 7])
        classes = rng.choice([1, 3, 20])
        n = rng.randint(0, 8)
        labels = []
        for _ in range(n):
            w = round(rng.uniform(0.005, 0.5), 3)
            h = round(rng.uniform(0.005, 0.5), 3)
            x = round(rng.uniform(0.0, 1.0), 3)
            y = round(rng.uniform(0.0, 1.0), 3)
            labels.append([rng.randrange(classes), x, y, w, h])
        jitter = [round(rng.uniform(-0.2, 0.2), 3), round(rng.uniform(-0.2, 0.2), 3),
                  round(rng.uniform(0.8, 1.2), 3), round(rng.uniform(0.8, 1.2), 3)]
        flip = rng.random() < 0.5
        _compare(solve, side, classes, labels, jitter, flip)


def check_invalid_inputs(solve):
    for args in [
        (0, 20, []),                          # side < 1
        (7, 0, []),                           # classes < 1
        (7, 20, [], [0.0, 0.0, 1.0]),         # jitter 长度 3
        (7, 20, [], None, 1),                 # flip 非布尔
        (7, 20, [[0, 0.5, 0.5, 0.2]]),        # label 长度 4
        (7, 20, [[20, 0.5, 0.5, 0.2, 0.2]]),  # id 越界
        (7, 20, [[-1, 0.5, 0.5, 0.2, 0.2]]),  # id 负
        (7, 20, [[0, 0.5, 0.5, 0.0, 0.2]]),   # w <= 0
        (7, 20, [[0, 0.5, 0.5, 0.2, -0.1]]),  # h <= 0
        (7, 20, [], [0.0, float("nan"), 1.0, 1.0]),  # jitter 非有限
    ]:
        try:
            solve(*args)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError(f"expected error: {args}")
