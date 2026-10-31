"""DIoU-NMS 抑制判据账（test cases）。"""

import math
import random


def _py_iou(a, b):
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    u = aa + ab - inter
    return inter / u if u > 0 else 0.0


def _py_diou(a, b):
    ca = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
    cb = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    rho2 = (ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2
    c2 = (max(a[2], b[2]) - min(a[0], b[0])) ** 2 + \
         (max(a[3], b[3]) - min(a[1], b[1])) ** 2
    return _py_iou(a, b) - rho2 / c2 if c2 > 0 else _py_iou(a, b)


def _py(boxes, scores, thresh):
    order = sorted(range(len(boxes)), key=lambda i: (-scores[i], i))
    keep, sup = [], {}
    alive = list(order)
    while alive:
        cur = alive.pop(0)
        keep.append(cur)
        nxt = []
        for j in alive:
            if _py_diou(boxes[cur], boxes[j]) > thresh:
                sup[j] = cur
            else:
                nxt.append(j)
        alive = nxt
    return {"keep": keep, "suppressed_by": sup}


def _close(actual, expect):
    if isinstance(actual, dict):
        assert set(actual.keys()) == set(expect.keys()), \
            f"{actual.keys()} vs {expect.keys()}"
        return all(_close(actual[k], expect[k]) for k in expect)
    if isinstance(actual, list):
        assert len(actual) == len(expect)
        return all(_close(a, e) for a, e in zip(actual, expect))
    assert abs(actual - expect) < 1e-9 if isinstance(expect, float) \
        else actual == expect, f"{actual} != {expect}"
    return True


def _compare(solve, boxes, scores, thresh):
    led = solve(boxes, scores, thresh)
    _close(led, _py(boxes, scores, thresh))
    return led


def check_anchor_diou_value(solve):
    # DIoU 公式锚点：A=[0,0,2,2], B=[1,1,3,3]
    # iou = 1/7；ρ²=2；外接 [0,0,3,3] c²=18 → diou = 1/7 − 1/9 = 2/63
    led = solve([[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0]],
                [0.9, 0.8], 2 / 63)
    # 阈值恰等于 2/63：严格大于不抑制
    assert led["keep"] == [0, 1]
    led2 = solve([[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0]],
                 [0.9, 0.8], 0.03)
    assert led2["keep"] == [0] and led2["suppressed_by"] == {1: 0}


def check_same_center_penalty_vanishes(solve):
    # 小位移重复检测：ρ²=0.04，外接 [0,0,2.2,2] c²=8.84
    # iou = 3.6/4.4；diou = 3.6/4.4 − 0.04/8.84 > 0.5 → 抑制
    led = solve([[0.0, 0.0, 2.0, 2.0], [0.2, 0.0, 2.2, 2.0]],
                [0.9, 0.8], 0.5)
    assert led["keep"] == [0] and led["suppressed_by"] == {1: 0}


def check_disjoint_never_suppressed(solve):
    # 完全分离：IoU=0，DIoU = −ρ²/c² ≤ 0 < 0.5 → 不抑制
    led = solve([[0.0, 0.0, 1.0, 1.0], [5.0, 5.0, 6.0, 6.0]],
                [0.9, 0.8], 0.5)
    assert led["keep"] == [0, 1] and led["suppressed_by"] == {}


def check_identical_boxes(solve):
    # 完全重合：ρ=0 → DIoU = IoU = 1 > 0.5 抑制
    led = solve([[1.0, 1.0, 4.0, 4.0], [1.0, 1.0, 4.0, 4.0]],
                [0.9, 0.8], 0.5)
    assert led["keep"] == [0] and led["suppressed_by"] == {1: 0}


def check_diou_catches_iou_miss(solve):
    # DIoU 核心卖点：IoU 低（不重叠）但中心近 → 判据换后仍不抑制（≤0）；
    # 反向对照：IoU 高会抑制，换 DIoU 后因中心远而不抑制
    # A=[0,0,2,2] B=[3,3,5,5]：IoU=0、中心距²=8、外接 25 → diou=−0.32
    led = solve([[0.0, 0.0, 2.0, 2.0], [3.0, 3.0, 5.0, 5.0]],
                [0.9, 0.8], 0.5)
    assert led["keep"] == [0, 1]
    # 对照：同样两框若用 IoU>0 判据也不抑制——改用大重叠但中心远的框
    # A=[0,0,10,10] B=[8,8,12,12]：inter=4, union=116 → iou≈0.0345
    # ρ²=50, 外接 [0,0,12,12] c²=288 → diou≈0.0345−0.1736<0
    led2 = solve([[0.0, 0.0, 10.0, 10.0], [8.0, 8.0, 12.0, 12.0]],
                 [0.9, 0.8], 0.03)
    # IoU 0.0345 > 0.03 会抑制，DIoU 0.0345−0.1736 = −0.139 不抑制
    assert led2["keep"] == [0, 1]


def check_score_order_stability(solve):
    # 分数并列：稳定排序取小下标先保留
    led = solve([[0.0, 0.0, 2.0, 2.0], [0.1, 0.0, 2.1, 2.0]],
                [0.7, 0.7], 0.5)
    assert led["keep"] == [0] and led["suppressed_by"] == {1: 0}


def check_chain_suppression(solve):
    # 三框链：A-B 重叠、B-C 重叠、A-C 不重叠
    # A=[0,0,2,2] B=[1,0,3,2] C=[2,0,4,2]：逐级由保留者抑制
    boxes = [[0.0, 0.0, 2.0, 2.0], [1.0, 0.0, 3.0, 2.0], [2.0, 0.0, 4.0, 2.0]]
    led = _compare(solve, boxes, [0.9, 0.8, 0.7], 0.3)
    # iou(A,B)=1/3, ρ²=1, c²=(3²+2²)=13 → diou≈0.3333−0.0769=0.2564<0.3
    # 全链不抑制（diou 阈值 0.3 下）
    assert led["keep"] == [0, 1, 2]


def check_oracle_random(solve):
    rng = random.Random(679)
    for _ in range(200):
        n = rng.randint(1, 12)
        boxes = []
        for _ in range(n):
            x1 = round(rng.uniform(0, 8), 2)
            y1 = round(rng.uniform(0, 8), 2)
            boxes.append([x1, y1, round(x1 + rng.uniform(0.5, 6), 2),
                          round(y1 + rng.uniform(0.5, 6), 2)])
        scores = [round(rng.random(), 3) for _ in range(n)]
        thresh = rng.choice([0.0, 0.3, 0.5, 0.7])
        _compare(solve, boxes, scores, thresh)


def check_invalid_inputs(solve):
    for args in [
        ([], [], 0.5),                          # 空
        ([[0.0, 0.0, 2.0, 2.0]], [], 0.5),      # 分数长度不符
        ([[2.0, 0.0, 1.0, 2.0]], [0.5], 0.5),   # x2 < x1
        ([[0.0, 0.0, 2.0, float('nan')]], [0.5], 0.5),
        ([[0.0, 0.0, 2.0, 2.0]], [float('inf')], 0.5),
        ([[0.0, 0.0, 2.0, 2.0]], [0.5], 1.5),   # 阈值越界
    ]:
        try:
            solve(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError: {args}")
