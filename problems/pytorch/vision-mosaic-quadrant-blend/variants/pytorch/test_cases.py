"""YOLOv4 Mosaic truth 侧账（test cases）。

独立 oracle：结构不同的直算实现，与 solution 互证。
"""

import math
import random


def _py_solve(w, h, cut_x, cut_y, i_mixup, shifts, old_truths, mosaic_bound):
    ls, rs, ts, bs = shifts
    dx = (cut_x - ls) / w if i_mixup in (1, 3) else -(w - cut_x - rs) / w
    dy = (cut_y - ts) / h if i_mixup in (2, 3) else -(h - cut_y - bs) / h
    kept, dropped = [], []
    for idx, t in enumerate(old_truths):
        xb, yb, wb, hb = t[0] + dx, t[1] + dy, t[2], t[3]
        l, r = int((xb - wb / 2) * w), int((xb + wb / 2) * w)
        tp, bt = int((yb - hb / 2) * h), int((yb + hb / 2) * h)
        if mosaic_bound:
            lb = cut_x if i_mixup in (1, 3) else 0
            rb = cut_x if i_mixup in (0, 2) else w
            tb = cut_y if i_mixup in (2, 3) else 0
            bb = cut_y if i_mixup in (0, 1) else h
            l, r = max(l, lb), min(r, rb)
            tp, bt = max(tp, tb), min(bt, bb)
            xb, wb = (r + l) / 2 / w, (r - l) / w
            yb, hb = (bt + tp) / 2 / h, (bt - tp) / h
        else:
            if l < 0:
                d = l / w
                xb, wb = xb - d / 2, wb + d
            if r > w:
                d = (r - w) / w
                xb, wb = xb - d / 2, wb - d
            if tp < 0:
                d = tp / h
                yb, hb = yb - d / 2, hb + d
            if bt > h:
                d = (bt - h) / h
                yb, hb = yb - d / 2, hb - d
            l, r = int((xb - wb / 2) * w), int((xb + wb / 2) * w)
            tp, bt = int((yb - hb / 2) * h), int((yb + hb / 2) * h)
        if (l >= 0 and r <= w and tp >= 0 and bt <= h and 0 < wb < 1
                and 0 < hb < 1 and 0 < xb < 1 and 0 < yb < 1
                and wb > 1 / w and hb > 1 / h):
            kept.append([xb, yb, wb, hb, t[4]])
        else:
            dropped.append(idx)
    if i_mixup == 0:
        plan = {"dst": [0, 0], "src": [w - cut_x - rs, h - cut_y - bs],
                "size": [cut_x, cut_y]}
    elif i_mixup == 1:
        plan = {"dst": [cut_x, 0], "src": [ls, h - cut_y - bs],
                "size": [w - cut_x, cut_y]}
    elif i_mixup == 2:
        plan = {"dst": [0, cut_y], "src": [w - cut_x - rs, ts],
                "size": [cut_x, h - cut_y]}
    else:
        plan = {"dst": [cut_x, cut_y], "src": [ls, ts],
                "size": [w - cut_x, h - cut_y]}
    return {"boxes": kept, "dropped_indices": dropped, "copy_plan": plan}


def _close(actual, expect, tol=1e-9):
    if isinstance(expect, dict):
        assert isinstance(actual, dict) and len(actual) == len(expect)
        return all(_close(actual[k], expect[k], tol) for k in expect)
    if isinstance(expect, list):
        assert isinstance(actual, list) and len(actual) == len(expect), \
            f"{actual!r} vs {expect!r}"
        return all(_close(a, e, tol) for a, e in zip(actual, expect))
    if isinstance(expect, float):
        assert abs(actual - expect) < tol, f"{actual} != {expect}"
        return True
    assert actual == expect, f"{actual!r} != {expect!r}"
    return True


def _compare(solve, w, h, cut_x, cut_y, i_mixup, shifts, old_truths,
             mosaic_bound=False):
    led = solve(w, h, cut_x, cut_y, i_mixup, shifts, old_truths, mosaic_bound)
    _close(led, _py_solve(w, h, cut_x, cut_y, i_mixup, shifts, old_truths,
                          mosaic_bound))
    return led


W, H = 416, 416


def check_no_shift_identity(solve):
    # shifts=0、cut 为半：象限 1 平移 xb += (cut_x)/w, yb −= (h−cut_y)/h
    led = _compare(solve, W, H, 208, 208, 1, [0, 0, 0, 0],
                   [[0.3, 0.6, 0.1, 0.1, 7]])
    b = led["boxes"][0]
    _close(b[0], 0.3 + 0.5)     # cut_x/416 = 0.5
    _close(b[1], 0.6 - 0.5)
    _close(b[2], 0.1)
    assert b[4] == 7            # id 透传
    assert led["dropped_indices"] == []
    # copy_plan：象限 1
    _close(led["copy_plan"]["dst"], [208, 0])
    _close(led["copy_plan"]["src"], [0, 208])
    _close(led["copy_plan"]["size"], [208, 208])


def check_quadrant_directions(solve):
    # 四象限平移方向：中心 (0,0)/(1,0)/(0,1)/(1,1) 全部被非 bound 收缩
    # 修正救回（left/top<0 → 中心抬进界内；right/bot>416 → 收回 416）
    t = [0.5, 0.5, 0.1, 0.1, 0]
    exp = {0: (0.02403846153846154, 0.02403846153846154),
           1: (1.0 - 20 / 832, 20 / 832),
           2: (20 / 832, 1.0 - 20 / 832),
           3: (1.0 - 20 / 832, 1.0 - 20 / 832)}
    for q in range(4):
        led = _compare(solve, W, H, 208, 208, q, [0, 0, 0, 0], [t])
        assert len(led["boxes"]) == 1, f"q={q} dropped"
        _close(led["boxes"][0][0], exp[q][0])
        _close(led["boxes"][0][1], exp[q][1])
        _close(led["boxes"][0][2], 0.1 - 20 / 416)   # |diff| 收缩
    # 象限 0 的 box 中心恰在 (0,0) 被修正到 (20/832, 20/832)——四象限
    # 落位关于图中心对称，方向账与修正账同时受检


def check_shift_compensation(solve):
    # shift 抵消部分平移：i_mixup=0, right_shift=26 → xb 平移量减小 26/416
    led = _compare(solve, W, H, 208, 208, 0, [0, 26, 0, 26],
                   [[0.5, 0.5, 0.1, 0.1, 3]])
    b = led["boxes"][0]
    _close(b[0], 0.5 - (416 - 208 - 26) / 416)     # −182/416
    _close(b[1], 0.5 - (416 - 208 - 26) / 416)


def check_bound_clamp_recompute(solve):
    # mosaic_bound：越出象限界的部分被 clamp 后角重算
    # 象限 0（界 x∈[0,208], y∈[0,208]），truth 中心被平移到左上区域
    led = _compare(solve, W, H, 208, 208, 0, [0, 0, 0, 0],
                   [[0.2, 0.2, 0.3, 0.3, 1]], mosaic_bound=True)
    # xb=0.2−0.5=−0.3 → left=(−0.45)·416≈−187 → clamp 0；right=(−0.15)·416≈−62
    # → clamp? right>rb? no, right<0 不 clamp → right=−62 <0 → 重算 wb=−62/416<0
    # → 终门 wb>0 失败 → drop
    assert led["dropped_indices"] == [0]
    # 大框横跨象限：clamp 后中心/宽由角还原
    led = _compare(solve, W, H, 208, 208, 0, [0, 0, 0, 0],
                   [[0.5, 0.5, 0.8, 0.8, 1]], mosaic_bound=True)
    # xb=0 → left=(0.5−0.4)·416=41… 平移后 xb=0, yb=0：
    # left=(0−0.4)·416=−166→0; right=(0+0.4)·416=166（≤208 不 clamp）
    # top 同 → 重算 xb=166/2/416, wb=166/416
    b = led["boxes"][0]
    _close(b[0], 166 / 2 / 416)
    _close(b[2], 166 / 416)


def check_expand_quirk(solve):
    # 非 bound：left<0 → diff<0 → wb+=diff 收缩（左缘裁到 0）、中心右移
    # truth 平移后 xb=−0.05、yb=0.4（yb 不触界）
    led = _compare(solve, W, H, 208, 208, 0, [0, 0, 0, 0],
                   [[0.45, 0.9, 0.2, 0.1, 2]])
    b = led["boxes"][0]
    _close(b[2], 0.2 - 62 / 416)          # left=−62 → |diff| 收缩
    _close(b[0], -0.05 + 62 / 832)
    _close(b[1], 0.4)                     # top=145≥0 不修正
    _close(b[3], 0.1)
    assert led["dropped_indices"] == []


def check_final_gate(solve):
    # lowest 门（wb>1/w）与中心半开门；象限 1 平移 (xb,yb)+=(0.5,−0.5)
    tiny = 0.5 / 416
    led = _compare(solve, W, H, 208, 208, 1, [0, 0, 0, 0],
                   [[0.3, 0.7, tiny, 0.1, 0]])
    # xb=0.8、wb=0.0012 < 1/416 → lowest 门失败
    assert led["dropped_indices"] == [0]
    led = _compare(solve, W, H, 208, 208, 1, [0, 0, 0, 0],
                   [[0.3, 0.7, 2.0 / 416, 0.1, 0]])
    assert led["dropped_indices"] == []
    # wb 恰 == 1（框满宽且全在界内）：wb<1 门失败 → drop（wb<=1 变异抓点）
    led = _compare(solve, W, H, 208, 208, 1, [0, 0, 0, 0],
                   [[0.0, 0.7, 1.0, 0.1, 0]])
    # xb=0.5、left=0、right=416 → 像素界过，仅 wb<1 失败
    assert led["dropped_indices"] == [0]


def check_pixel_int_truncation(solve):
    # int 截断 vs 浮点：right=(xb+wb/2)·w 恰为 416.0 → int 截断仍 ≤w 过门
    led = _compare(solve, W, H, 208, 208, 1, [0, 0, 0, 0],
                   [[0.49, 0.6, 0.01, 0.1, 5]])
    # xb=0.99、right=int((0.99+0.005)·416)=int(413.9)=413 ≤ 416 ✓
    assert led["dropped_indices"] == []
    _close(led["boxes"][0][0], 0.99)
    _close(led["boxes"][0][2], 0.01)


def check_copy_plan_quadrants(solve):
    base = dict(cut_x=100, cut_y=300, shifts=[4, 6, 8, 10])
    led = _compare(solve, W, H, 100, 300, 0, [4, 6, 8, 10], [])
    _close(led["copy_plan"],
           {"dst": [0, 0], "src": [W - 100 - 6, H - 300 - 10],
            "size": [100, 300]})
    led = _compare(solve, W, H, 100, 300, 1, [4, 6, 8, 10], [])
    _close(led["copy_plan"],
           {"dst": [100, 0], "src": [4, H - 300 - 10],
            "size": [W - 100, 300]})
    led = _compare(solve, W, H, 100, 300, 2, [4, 6, 8, 10], [])
    _close(led["copy_plan"],
           {"dst": [0, 300], "src": [W - 100 - 6, 8], "size": [100, H - 300]})
    led = _compare(solve, W, H, 100, 300, 3, [4, 6, 8, 10], [])
    _close(led["copy_plan"],
           {"dst": [100, 300], "src": [4, 8], "size": [W - 100, H - 300]})


def check_oracle_random(solve):
    rng = random.Random(704)
    for _ in range(250):
        w = rng.choice([208, 320, 416])
        h = w
        cut_x = rng.randint(0, w)
        cut_y = rng.randint(0, h)
        q = rng.randrange(4)
        shifts = [rng.randint(0, max(0, cut_x)), rng.randint(0, max(0, w - cut_x)),
                  rng.randint(0, max(0, cut_y)), rng.randint(0, max(0, h - cut_y))]
        n = rng.randint(0, 6)
        truths = [[round(rng.uniform(0, 1), 3), round(rng.uniform(0, 1), 3),
                   round(rng.uniform(0.001, 0.6), 3),
                   round(rng.uniform(0.001, 0.6), 3), rng.randrange(5)]
                  for _ in range(n)]
        _compare(solve, w, h, cut_x, cut_y, q, shifts, truths,
                 mosaic_bound=rng.random() < 0.5)


def check_invalid_inputs(solve):
    good = dict(w=416, h=416, cut_x=208, cut_y=208, i_mixup=0,
                shifts=[0, 0, 0, 0], old_truths=[[0.5, 0.5, 0.1, 0.1, 0]])
    def call(**kw):
        a = {**good, **kw}
        return solve(a["w"], a["h"], a["cut_x"], a["cut_y"], a["i_mixup"],
                     a["shifts"], a["old_truths"])
    for kw in [
        dict(w=0), dict(h=-1), dict(w=1.5),
        dict(cut_x=-1), dict(cut_x=417), dict(cut_y=417),
        dict(i_mixup=4), dict(i_mixup=-1),
        dict(shifts=[0, 0, 0]), dict(shifts=[0, 0, 0, -1]),
        dict(old_truths=[[0.5, 0.5, 0.1, 0.1]]),
        dict(old_truths=[[0.5, 0.5, 0.0, 0.1, 0]]),
        dict(old_truths=[[float("inf"), 0.5, 0.1, 0.1, 0]]),
    ]:
        try:
            call(**kw)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError(f"expected error: {kw}")
