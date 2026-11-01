"""KL-Loss variance voting 账（test cases）。

独立 oracle：活跃列表 + 惰性删除结构（与官方 swap-tail 布局不同、
逐原始索引查 IoU），与 solution 互证。
"""

import math
import random


def _oracle(dets, confs, thresh, iou_sigma):
    dets = [[float(v) for v in row] for row in dets]
    confs = [[float(v) for v in row] for row in confs]
    N = len(dets)
    if thresh > 0.1:
        thresh = 0.01

    def _pair(a, b):
        iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]) + 1.0)
        ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]) + 1.0)
        inter = iw * ih
        aa = (a[2] - a[0] + 1.0) * (a[3] - a[1] + 1.0)
        ab = (b[2] - b[0] + 1.0) * (b[3] - b[1] + 1.0)
        return inter / (aa + ab - inter)

    iou = [[_pair(dets[i], dets[j]) for j in range(N)] for i in range(N)]

    # order = 官方槽位序的原索引；pick swap 与 tail-swap 都完整模拟
    # （平分时 argmax 取槽位序首个最大——swap 会改变该序，不可用 remove 近似）
    order = list(range(N))
    score = {i: dets[i][4] for i in order}
    box = {i: dets[i][:4] for i in order}
    steps, removed, final = [], [], []
    pos = 0
    while pos < len(order):
        best_j = pos
        for j in range(pos, len(order)):     # 槽位序首个最大
            if score[order[j]] > score[order[best_j]]:
                best_j = j
        pick = order[best_j]
        order[pos], order[best_j] = order[best_j], order[pos]
        voters = [order[k] for k in range(pos, len(order))
                  if iou[pick][order[k]] > thresh]
        nb = []
        for c in range(4):
            num = den = 0.0
            for k in voters:                 # 槽位序累加（与官方一致）
                p = math.exp(-(1.0 - iou[pick][k]) ** 2 / iou_sigma)
                num += p * box[k][c] / (confs[k][c] ** 2)
                den += p / (confs[k][c] ** 2)
            nb.append(num / den if den else box[pick][c])
        box[pick] = nb
        steps.append({
            "idx": pick,
            "box": nb[:],
            "voters": sorted(voters),
            "weights": [[math.exp(-(1.0 - iou[pick][k]) ** 2 / iou_sigma)
                         / (confs[k][c] ** 2) for c in range(4)]
                        for k in sorted(voters)],
        })
        final.append(nb + [score[pick]])
        i2 = pos + 1
        while i2 < len(order):               # 槽位序衰减，破线者 tail-swap
            k = order[i2]
            if iou[pick][k] > 0:
                score[k] *= math.exp(-(iou[pick][k] ** 2) / 0.5)
                if score[k] < 0.001:
                    removed.append(k)
                    order[i2], order[-1] = order[-1], order[i2]
                    order.pop()
                    i2 -= 1
            i2 += 1
        pos += 1
    return {"ious": iou, "steps": steps, "final": final, "removed": removed}


def _close(a, b, tol=1e-9):
    assert math.isclose(a, b, rel_tol=tol, abs_tol=tol), f"{a} != {b}"


def _compare(solve, dets, confs, thresh, iou_sigma):
    led = solve(dets, confs, thresh, iou_sigma)
    ref = _oracle(dets, confs, thresh, iou_sigma)
    N = len(dets)
    for i in range(N):
        for j in range(N):
            _close(led["ious"][i][j], ref["ious"][i][j])
    assert [s["idx"] for s in led["steps"]] == [s["idx"] for s in ref["steps"]]
    for s_led, s_ref in zip(led["steps"], ref["steps"]):
        assert s_led["voters"] == s_ref["voters"]
        for c in range(4):
            _close(s_led["box"][c], s_ref["box"][c])
            for w_led, w_ref in zip(s_led["weights"], s_ref["weights"]):
                _close(w_led[c], w_ref[c])
    assert len(led["final"]) == len(ref["final"])
    for f_led, f_ref in zip(led["final"], ref["final"]):
        for a, b in zip(f_led, f_ref):
            _close(a, b)
    assert led["removed"] == ref["removed"]
    return led


def check_two_boxes_vote(solve):
    # 相同分数结构不同 conf：融合坐标偏向 σ 小者（权重 1/σ²）
    dets = [[0.0, 0.0, 10.0, 10.0, 0.9], [2.0, 2.0, 12.0, 12.0, 0.7]]
    confs = [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]]
    led = _compare(solve, dets, confs, 0.3, 0.5)
    iou = led["ious"][0][1]
    _close(iou, 81.0 / 161.0)
    p = math.exp(-(1.0 - iou) ** 2 / 0.5)
    # A 融合（voters 含自身 + B）
    den = 1.0 + p / 4.0
    _close(led["steps"][0]["box"][0], (0.0 * 1.0 + 2.0 * p / 4.0) / den)
    _close(led["steps"][0]["box"][2], (10.0 * 1.0 + 12.0 * p / 4.0) / den)
    # B 被 A 衰减（iou>0 且 iou>thresh 也投过票，但分数只衰减不重投）
    _close(led["final"][1][4], 0.7 * math.exp(-(iou ** 2) / 0.5))
    # weights 逐坐标：B 的 weight = p/σ² = p/4
    _close(led["steps"][0]["weights"][1][0], p / 4.0)
    _close(led["steps"][0]["weights"][0][0], 1.0)


def check_thresh_clamp(solve):
    # thresh>0.1 钳到 0.01：与直接传 0.01 结果一致
    dets = [[0.0, 0.0, 10.0, 10.0, 0.9], [9.0, 0.0, 19.0, 10.0, 0.8],
            [40.0, 0.0, 49.0, 9.0, 0.5]]
    confs = [[1.0] * 4] * 3
    led_a = _compare(solve, dets, confs, 0.3, 0.5)
    led_b = _compare(solve, dets, confs, 0.01, 0.5)
    assert [s["voters"] for s in led_a["steps"]] == \
        [s["voters"] for s in led_b["steps"]]
    # 中间框 [9,0,19,10] 与 pick [0,0,10,10] 的 +1 约定 IoU
    # inter=2*11=22, areas 121+121 → 22/220=0.1 > 0.01 参与投票
    # 而 [40,...] 与谁都不重叠；第 2 轮 pick B 时 C 的 IoU=0 → 只有自身
    assert led_a["steps"][0]["voters"] == [0, 1]
    assert led_a["steps"][1]["voters"] == [1]
    # thresh=0.5（>0.1 仍钳 0.01）→ 不变
    led_c = _compare(solve, dets, confs, 0.5, 0.5)
    assert [s["voters"] for s in led_c["steps"]] == \
        [s["voters"] for s in led_a["steps"]]


def check_suppression_without_vote(solve):
    # iou ∈ (0, thresh)：不投票但仍衰减（衰减门是 iou>0 非 thresh）
    dets = [[0.0, 0.0, 199.0, 99.0, 0.9], [199.0, 0.0, 398.0, 99.0, 0.7]]
    confs = [[1.0] * 4] * 2
    led = _compare(solve, dets, confs, 0.01, 0.5)
    iou = led["ious"][0][1]
    assert 0 < iou < 0.01
    assert led["steps"][0]["voters"] == [0]          # 自身才投票
    _close(led["final"][1][4], 0.7 * math.exp(-(iou ** 2) / 0.5))
    assert led["final"][1][4] < 0.7


def check_score_floor_tail_swap(solve):
    # 5 个重复框：链式衰减把后续分数压破 0.001 → 移除
    dets = [[0.0, 0.0, 10.0, 10.0, s] for s in
            [0.9, 0.5, 0.5, 0.5, 0.5]]
    confs = [[1.0] * 4] * 5
    led = _compare(solve, dets, confs, 0.3, 0.5)
    assert len(led["final"]) == 4
    assert led["removed"] == [4]     # 第 4 轮衰减时破 0.001 的那个
    # 第 2 轮 pick 的分数已衰减 1 次
    _close(led["final"][1][4], 0.5 * math.exp(-2.0))


def check_decay_changes_pick_order(solve):
    # 分数衰减重排 pick 顺序：0.8 的框被 0.9 框压到 0.108 < 0.5
    dets = [[0.0, 0.0, 10.0, 10.0, 0.9], [0.0, 0.0, 10.0, 10.0, 0.8],
            [50.0, 50.0, 60.0, 60.0, 0.5]]
    confs = [[1.0] * 4] * 3
    led = _compare(solve, dets, confs, 0.3, 0.5)
    assert [s["idx"] for s in led["steps"]] == [0, 2, 1]
    _close(led["final"][2][4], 0.8 * math.exp(-2.0))


def check_stale_iou_after_fusion(solve):
    # 已 pick 的框出列后不再参与投票：第 2 轮 pick 的 voters 只有自身，
    # 即便它与第 1 轮 pick 的旧 IoU 仍超 thresh（官方一次出列制）
    dets = [[10.0, 0.0, 20.0, 10.0, 0.9], [0.0, 0.0, 10.0, 10.0, 0.8]]
    confs = [[5.0, 5.0, 5.0, 5.0], [0.5, 0.5, 0.5, 0.5]]
    led = _compare(solve, dets, confs, 0.01, 0.5)
    assert led["steps"][0]["voters"] == [0, 1]
    assert led["steps"][1]["voters"] == [1]
    # pick A 被 σ 小的 B 拉近：x1 从 10 拉向 0、x2 从 20 拉向 10
    assert led["steps"][0]["box"][0] < 10.0
    assert led["steps"][0]["box"][2] < 20.0
    # B 自身融合时 voters 只有自己 → 坐标不变
    _close(led["steps"][1]["box"][0], 0.0)
    _close(led["steps"][1]["box"][2], 10.0)


def check_single_box_per_coord_sigma(solve):
    # 单框：p=1，融合 = 自身（各坐标 σ 不同也逐坐标还原）
    led = _compare(solve, [[3.0, 4.0, 30.0, 40.0, 0.6]],
                   [[2.0, 0.5, 1.0, 4.0]], 0.3, 0.5)
    _close(led["final"][0][0], 3.0)
    _close(led["final"][0][1], 4.0)
    _close(led["final"][0][2], 30.0)
    _close(led["final"][0][3], 40.0)
    _close(led["final"][0][4], 0.6)
    assert led["steps"][0]["voters"] == [0]
    _close(led["steps"][0]["weights"][0][0], 1.0 / 4.0)
    _close(led["steps"][0]["weights"][0][1], 1.0 / 0.25)
    # 空输入：不报错，全空 ledger
    led = _compare(solve, [], [], 0.3, 0.5)
    assert led["steps"] == [] and led["final"] == [] and led["removed"] == []


def check_random_against_oracle(solve):
    rng = random.Random(705)
    for _ in range(200):
        n = rng.randrange(1, 9)
        xs = sorted(round(rng.uniform(0, 100), 1) for _ in range(2 * n))
        dets, confs = [], []
        for i in range(n):
            x1, x2 = xs[2 * i], xs[2 * i + 1]
            y1 = round(rng.uniform(0, 80), 1)
            h = round(rng.uniform(5, 60), 1)
            dets.append([x1, y1, x2, y1 + h,
                         round(rng.uniform(0.05, 0.99), 3)])
            confs.append([round(rng.uniform(0.5, 3.0), 2) for _ in range(4)])
        thresh = rng.choice([0.01, 0.05, 0.3, 0.5, 0.2])
        iou_sigma = rng.choice([0.1, 0.5, 1.0])
        _compare(solve, dets, confs, thresh, iou_sigma)


def check_invalid_inputs(solve):
    good = dict(dets=[[0.0, 0.0, 10.0, 10.0, 0.9]],
                confs=[[1.0, 1.0, 1.0, 1.0]], thresh=0.3, iou_sigma=0.5)

    def call(**kw):
        a = {**good, **kw}
        return solve(a["dets"], a["confs"], a["thresh"], a["iou_sigma"])
    for kw in [
        dict(dets=[[0.0, 0.0, 10.0, 10.0]]),              # 缺分数
        dict(dets=[[0.0, 0.0, 10.0, 10.0, 0.9, 1.0]]),    # 多一列
        dict(dets=[[0.0, 0.0, 10.0, 10.0, 0.9],
                   [0.0, 0.0, 10.0, 10.0]]),              # 行长不齐
        dict(confs=[[1.0, 1.0, 1.0]]),                    # σ 缺坐标
        dict(confs=[[1.0, 1.0, 1.0, 0.0]]),               # σ=0
        dict(confs=[[1.0, 1.0, 1.0, -1.0]]),              # σ<0
        dict(iou_sigma=0.0),
        dict(iou_sigma=-1.0),
        dict(thresh=float("nan")),
    ]:
        try:
            call(**kw)
        except (ValueError, TypeError, ZeroDivisionError):
            pass
        else:
            raise AssertionError(f"expected error: {kw}")
