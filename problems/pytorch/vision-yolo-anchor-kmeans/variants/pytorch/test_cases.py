"""YOLOv2 anchor k-means（1−IoU）账（test cases）。"""

import math
import random


def _py_iou(b, c):
    w1, h1 = b
    w2, h2 = c
    if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
        return 0.0
    inter = min(w1, w2) * min(h1, h2)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union


def _py(boxes, k, init):
    """独立 oracle：逐步重放官方 C 循环。"""
    centers = [[float(boxes[i][0]), float(boxes[i][1])] for i in init]
    assigns = [0] * len(boxes)

    def closest(box):
        best, bd = 0, 1.0 - _py_iou(box, centers[0])
        for j in range(1, k):
            nd = 1.0 - _py_iou(box, centers[j])
            if nd < bd:
                bd, best = nd, j
        return best

    def E():
        conv = True
        for i in range(len(boxes)):
            c = closest(boxes[i])
            if c != assigns[i]:
                conv = False
            assigns[i] = c
        return conv

    def M():
        cnt = [0] * k
        acc = [[0.0, 0.0] for _ in range(k)]
        for i in range(len(boxes)):
            cnt[assigns[i]] += 1
            acc[assigns[i]][0] += boxes[i][0]
            acc[assigns[i]][1] += boxes[i][1]
        for j in range(k):
            if cnt[j]:
                centers[j] = [acc[j][0] / cnt[j], acc[j][1] / cnt[j]]
            else:
                centers[j] = [0.0, 0.0]

    if k == 1:
        M()
    while not E():
        M()
    ious = [_py_iou(boxes[i], centers[assigns[i]]) for i in range(len(boxes))]
    return {"anchors": [list(c) for c in centers],
            "assignments": list(assigns),
            "avg_iou": sum(ious) / len(boxes)}


def _close(actual, expect, tol=1e-9):
    if isinstance(actual, dict):
        for kk in expect:
            assert _close(actual[kk], expect[kk], tol), f"mismatch at {kk}"
        return True
    if isinstance(actual, list):
        assert len(actual) == len(expect), f"{actual} vs {expect}"
        return all(_close(a, e, tol) for a, e in zip(actual, expect))
    if isinstance(expect, float):
        assert abs(actual - expect) < tol, f"{actual} != {expect}"
        return True
    assert actual == expect, f"{actual!r} != {expect!r}"
    return True


def _compare(solve, boxes, k, init):
    led = solve(boxes, k, init)
    ref = _py(boxes, k, init)
    _close(led, ref)
    return led


def check_anchor_two_clusters(solve):
    # 手算：k=2, init=[0,2] → anchors [[1,1.5],[10,10]]，assignments [0,0,1]
    # avg_iou = (2/3 + 3/4 + 1)/3
    boxes = [[1.0, 1.0], [1.0, 2.0], [10.0, 10.0]]
    led = _compare(solve, boxes, 2, [0, 2])
    _close(led["anchors"], [[1.0, 1.5], [10.0, 10.0]])
    assert led["assignments"] == [0, 0, 1]
    _close(led["avg_iou"], (2 / 3 + 3 / 4 + 1) / 3)


def check_empty_cluster_quirk(solve):
    # 两点两簇同一下标初始化：b1 并列取 0 → cluster1 空
    # E 步 converged（assignments 从全 0 起）→ centers 不变，quirk 未触发
    boxes = [[1.0, 1.0], [2.0, 2.0]]
    led = _compare(solve, boxes, 2, [0, 0])
    _close(led["anchors"], [[1.0, 1.0], [1.0, 1.0]])
    assert led["assignments"] == [0, 0]
    _close(led["avg_iou"], (1.0 + 0.25) / 2)


def check_empty_cluster_triggered(solve):
    # k=3、init=[0,1,1]：首 E 把 b1 分到 cluster1（≠初始 0）→ M 步
    # cluster2 无成员 → (0,0) quirk（不重采样）
    boxes = [[1.0, 1.0], [10.0, 10.0]]
    led = _compare(solve, boxes, 3, [0, 1, 1])
    _close(led["anchors"], [[1.0, 1.0], [10.0, 10.0], [0.0, 0.0]])
    assert led["assignments"] == [0, 1]
    _close(led["avg_iou"], 1.0)
    # 注：k=2 init=[0,0] 时 assignments 从全 0 起、首 E 即收敛，quirk 不触发


def check_k1_special_case(solve):
    # k=1：官方先跑 M 步（否则全 0 assignments 直接假收敛）
    boxes = [[2.0, 2.0], [4.0, 6.0]]
    led = _compare(solve, boxes, 1, [0])
    # M 步后 centroid = [3,4]；E 步全 0 收敛
    _close(led["anchors"], [[3.0, 4.0]])
    assert led["assignments"] == [0, 0]
    # iou(b0,[3,4]) = 4/(4+12−4)=1/3；iou(b1)=12/(24+12−12)=1/2
    _close(led["avg_iou"], (1 / 3 + 1 / 2) / 2)


def check_tie_lower_index(solve):
    # 对称中心并列：closest 取小下标
    boxes = [[2.0, 2.0]]
    led = _compare(solve, boxes, 2, [0, 0])
    assert led["assignments"] == [0]


def check_iou_wh_boundary(solve):
    # 经 solve 间接钉 IoU 公式：k=1 单点收敛后 avg_iou = iou(box, centroid)
    led = solve([[2.0, 3.0]], 1, [0])
    assert led["avg_iou"] == 1.0                     # iou(x,x)=1
    led = solve([[1.0, 1.0], [3.0, 1.0]], 1, [0])
    # k=1 特判 M 步 → centroid=[2,1]；iou([1,1],[2,1])=1/2、iou([3,1],[2,1])=2/3
    assert abs(led["avg_iou"] - (1 / 2 + 2 / 3) / 2) < 1e-12
    try:
        solve([[0.0, 1.0]], 1, [0])                  # 非正维 → 非法入参
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def check_oracle_random(solve):
    rng = random.Random(678)
    for _ in range(150):
        n = rng.randint(2, 12)
        k = rng.randint(1, min(5, n))
        boxes = [[round(rng.uniform(0.5, 30), 3), round(rng.uniform(0.5, 30), 3)]
                 for _ in range(n)]
        init = [rng.randrange(n) for _ in range(k)]
        _compare(solve, boxes, k, init)


def check_invalid_inputs(solve):
    for args in [
        ([], 1, [0]),                       # 空 boxes
        ([[1.0]], 1, [0]),                  # 非二元
        ([[0.0, 1.0]], 1, [0]),             # 非正宽
        ([[1.0, 1.0]], 0, []),              # k < 1
        ([[1.0, 1.0]], 2, [0]),             # init 长度不符
        ([[1.0, 1.0]], 1, [1]),             # init 下标越界
    ]:
        try:
            solve(*args)
        except (ValueError, IndexError):
            pass
        else:
            raise AssertionError(f"expected error: {args}")
