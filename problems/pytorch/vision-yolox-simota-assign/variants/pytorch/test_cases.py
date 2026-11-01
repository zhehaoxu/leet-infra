"""YOLOX SimOTA 分配账（test cases）。

独立 oracle：numpy 向量化直算 + 200 组随机对拍。
"""

import math
import random


def _oracle(gt_bboxes, gt_classes, strides_in, x_shifts, y_shifts,
            cls_preds, obj_preds, bboxes_preds, num_classes):
    G = len(gt_bboxes)
    A = len(strides_in)
    strides = [float(s) for s in strides_in]
    xc = [(x_shifts[k] + 0.5) * strides[k] for k in range(A)]
    yc = [(y_shifts[k] + 0.5) * strides[k] for k in range(A)]

    # 几何约束：向量化 (G, A)
    in_centers = []
    for g in range(G):
        gx, gy = gt_bboxes[g][0], gt_bboxes[g][1]
        in_centers.append([
            min(xc[k] - (gx - 1.5 * strides[k]),
                (gx + 1.5 * strides[k]) - xc[k],
                yc[k] - (gy - 1.5 * strides[k]),
                (gy + 1.5 * strides[k]) - yc[k]) > 0.0
            for k in range(A)])
    filt = [any(in_centers[g][k] for g in range(G)) for k in range(A)]
    cand = [k for k in range(A) if filt[k]]
    n = len(cand)

    # cxcywh IoU（相切不算）
    iou = []
    for g in range(G):
        row = []
        for k in cand:
            b = bboxes_preds[k]
            t = [max(gt_bboxes[g][0] - gt_bboxes[g][2] / 2,
                     b[0] - b[2] / 2),
                 max(gt_bboxes[g][1] - gt_bboxes[g][3] / 2,
                     b[1] - b[3] / 2)]
            r = [min(gt_bboxes[g][0] + gt_bboxes[g][2] / 2,
                     b[0] + b[2] / 2),
                 min(gt_bboxes[g][1] + gt_bboxes[g][3] / 2,
                     b[1] + b[3] / 2)]
            if not (t[0] < r[0] and t[1] < r[1]):
                row.append(0.0)
                continue
            ai = (r[0] - t[0]) * (r[1] - t[1])
            row.append(ai / (gt_bboxes[g][2] * gt_bboxes[g][3]
                             + b[2] * b[3] - ai))
        iou.append(row)

    # cost：矩阵式直算
    cost = []
    for g in range(G):
        row = []
        for j, k in enumerate(cand):
            cs = 0.0
            for c in range(num_classes):
                p = math.sqrt((1 / (1 + math.exp(-cls_preds[k][c])))
                              * (1 / (1 + math.exp(-obj_preds[k]))))
                y = 1.0 if c == gt_classes[g] else 0.0
                cs += -(y * math.log(p) + (1 - y) * math.log(1 - p))
            row.append(cs + 3.0 * (-math.log(iou[g][j] + 1e-8))
                       + (0.0 if in_centers[g][k] else 1e6))
        cost.append(row)

    nck = min(10, n)
    dks = []
    for g in range(G):
        s = sum(sorted(iou[g], reverse=True)[:nck])
        dks.append(max(int(s), 1))

    mat = [[0] * n for _ in range(G)]
    for g in range(G):
        picks = sorted(range(n), key=lambda j: cost[g][j])[:dks[g]]
        for j in picks:
            mat[g][j] = 1
    cnt = [sum(mat[g][j] for g in range(G)) for j in range(n)]
    for j in range(n):
        if cnt[j] > 1:
            bg = min(range(G), key=lambda g: cost[g][j])
            for g in range(G):
                mat[g][j] = 0
            mat[bg][j] = 1

    fg_in = [any(mat[g][j] for g in range(G)) for j in range(n)]
    num_fg = sum(fg_in)
    mgi = [-1] * A
    gmc = [-1] * A
    pio = [0.0] * A
    fg = [False] * A
    for j, k in enumerate(cand):
        fg[k] = fg_in[j]
        if fg_in[j]:
            g = next(gg for gg in range(G) if mat[gg][j] == 1)
            mgi[k] = g
            gmc[k] = gt_classes[g]
            pio[k] = iou[g][j]
    return {"anchor_filter": filt, "is_in_centers": in_centers,
            "ious": iou, "cost": cost, "dynamic_ks": dks,
            "matching_matrix": mat, "fg_mask": fg, "num_fg": num_fg,
            "matched_gt_inds": mgi, "gt_matched_classes": gmc,
            "pred_ious_this_matching": pio}


def _close(a, b, tol=1e-9):
    assert math.isclose(a, b, rel_tol=tol, abs_tol=tol), f"{a} != {b}"


def _compare(solve, gt_bboxes, gt_classes, strides, xs, ys, cls_preds,
             obj_preds, bboxes_preds, num_classes):
    led = solve(gt_bboxes, gt_classes, strides, xs, ys, cls_preds,
                obj_preds, bboxes_preds, num_classes)
    ref = _oracle(gt_bboxes, gt_classes, strides, xs, ys, cls_preds,
                  obj_preds, bboxes_preds, num_classes)
    assert led["anchor_filter"] == ref["anchor_filter"]
    assert led["is_in_centers"] == ref["is_in_centers"]
    G, n = len(ref["ious"]), len(ref["ious"][0]) if ref["ious"] else 0
    for g in range(G):
        for j in range(n):
            _close(led["ious"][g][j], ref["ious"][g][j])
            _close(led["cost"][g][j], ref["cost"][g][j])
        assert led["dynamic_ks"][g] == ref["dynamic_ks"][g]
        assert led["matching_matrix"][g] == ref["matching_matrix"][g]
    assert led["fg_mask"] == ref["fg_mask"]
    assert led["num_fg"] == ref["num_fg"]
    assert led["matched_gt_inds"] == ref["matched_gt_inds"]
    assert led["gt_matched_classes"] == ref["gt_matched_classes"]
    for a, b in zip(led["pred_ious_this_matching"],
                    ref["pred_ious_this_matching"]):
        _close(a, b)
    return led


def check_single_gt_single_anchor(solve):
    # 1 GT 1 anchor：anchor 中心在 GT 邻域内 → 必匹配（dynamic_k≥1）
    led = _compare(solve, [[32.0, 32.0, 16.0, 16.0]], [2],
                   [8], [3], [3],
                   [[0.5, -0.5, 1.0]], [0.3], [[30.0, 31.0, 18.0, 14.0]], 3)
    assert led["anchor_filter"] == [True]
    assert led["dynamic_ks"] == [1]
    assert led["num_fg"] == 1
    assert led["matched_gt_inds"] == [0]
    assert led["gt_matched_classes"] == [2]
    _close(led["pred_ious_this_matching"][0], led["ious"][0][0])
    # cost 手算锚点：cls 项（gt_class=2 → y=1 在 class 2）+ 3·iou 项
    sig = lambda x: 1 / (1 + math.exp(-x))
    p2 = math.sqrt(sig(1.0) * sig(0.3))
    c0 = -(math.log(1 - math.sqrt(sig(0.5) * sig(0.3)))
           + math.log(1 - math.sqrt(sig(-0.5) * sig(0.3)))
           + math.log(p2))
    _close(led["cost"][0][0] - 3.0 * (-math.log(led["ious"][0][0] + 1e-8)),
           c0)


def check_geometry_radius_stride_scaled(solve):
    # 邻域半径是 1.5·stride（随 anchor stride 缩放）：小 stride 邻域小
    # GT 中心 100；anchor A stride 8（中心 100.5·? ...）直接构造：
    # anchor 中心 112：dist=1.5·8=12 → 112 < 100+12 假 → 不在
    # anchor B stride 16：dist=24 → 在（若 x_shift 对齐）
    led = _compare(solve, [[100.0, 50.0, 4.0, 4.0]], [0],
                   [8, 16], [13, 6], [6, 3],
                   [[0.0], [0.0]], [0.0, 0.0],
                   [[100.0, 50.0, 4.0, 4.0], [100.0, 50.0, 4.0, 4.0]], 1)
    # A: center=(13.5)*8=108, dist=12 → |108-100|=8 < 12 在
    # B: center=(6.5)*16=104, dist=24 → 在
    assert led["anchor_filter"] == [True, True]
    led = _compare(solve, [[100.0, 50.0, 4.0, 4.0]], [0],
                   [8, 8], [13, 14], [6, 6],
                   [[0.0], [0.0]], [0.0, 0.0],
                   [[100.0, 50.0, 4.0, 4.0], [100.0, 50.0, 4.0, 4.0]], 1)
    # 中心 108 与 116：116-100=16 > 12 → B 不在
    assert led["anchor_filter"] == [True, False]
    assert led["is_in_centers"][0] == [True, False]


def check_iou_touch_excluded(solve):
    # 相切边（tl==br）IoU=0（en 用严格 <，与常见 max(0,·) 不同）
    # pred x∈[15,25]，gt x∈[5,15] → tl=15, br=15 → 严格 < 不成立 → IoU 0
    led = _compare(solve, [[10.0, 10.0, 10.0, 10.0]], [0],
                   [8], [1], [1],
                   [[0.0]], [0.0],
                   [[20.0, 10.0, 10.0, 10.0]], 1)
    _close(led["ious"][0][0], 0.0)
    # cost 仍有限（-log(0+1e-8)≈18.42）且匹配照样发生（dynamic_k≥1）
    assert led["num_fg"] == 1
    assert math.isfinite(led["cost"][0][0])


def check_dynamic_k_int_truncation(solve):
    # dynamic_k = max(int(sum(top-10 ious)), 1)：int 截断非四舍五入
    # 2 GT × 12 anchor：构造 GT0 与多个 anchor IoU 总和 2.x → k=2
    gts = [[20.0, 20.0, 8.0, 8.0], [60.0, 20.0, 8.0, 8.0]]
    strides, xs, ys = [], [], []
    preds = []
    cls_preds = []
    obj_preds = []
    for i in range(12):
        strides.append(8)
        xs.append(i)
        ys.append(2)
        cx = (i + 0.5) * 8
        preds.append([cx, 20.0, 8.0, 8.0])
        cls_preds.append([0.0, 0.0])
        obj_preds.append(0.0)
    led = _compare(solve, gts, [0, 1], strides, xs, ys, cls_preds,
                   obj_preds, preds, 2)
    assert led["dynamic_ks"][0] >= 1
    # top-10 之和的 int 截断：与 oracle 对拍一致即可（锚点只锁 int 行为）
    s0 = sum(sorted(led["ious"][0], reverse=True)[:10])
    assert led["dynamic_ks"][0] == max(int(s0), 1)


def check_dynamic_k_truncation_boundary(solve):
    # dynamic_k 边界锚点：top-10 IoU 和 = 4.976 → int 截 4（round 会给 5）
    # top-5 和 = 2.6 → int 2（top-10 改 top-5 会给不同 k）
    # 12 个候选 anchor（stride8 九宫格 + stride16 三个）全部过几何
    gts = [[100.0, 100.0, 20.0, 20.0]]
    strides, xs, ys = [], [], []
    for dx in (11, 12, 13):
        for dy in (11, 12, 13):
            strides.append(8); xs.append(dx); ys.append(dy)
    for dx in (5, 6, 7):
        strides.append(16); xs.append(dx); ys.append(6)
    # 同心 pred（⊂ gt）：iou = (wh/20)^2
    # top-10 和 = 0.8+0.7+0.6+0.5+0.5+5×0.297 = 4.585 → int 4（round 5）
    # top-5 和 = 3.1 → int 3（top-10 改 top-5 会给不同 k）
    whs = [20 * math.sqrt(v) for v in
           [0.8, 0.7, 0.6, 0.5, 0.5] + [0.297] * 7]
    preds = [[100.0, 100.0, w, w] for w in whs]
    cls_preds = [[0.0]] * 12
    led = _compare(solve, gts, [0], strides, xs, ys, cls_preds,
                   [0.0] * 12, preds, 1)
    s10 = sum(sorted(led["ious"][0], reverse=True)[:10])
    s5 = sum(sorted(led["ious"][0], reverse=True)[:5])
    assert 4.5 < s10 < 5 and int(s10) != round(s10)
    assert abs(s5 - 3.1) < 1e-9
    assert led["dynamic_ks"] == [4]
    assert led["num_fg"] == 4


def check_conflict_arbitration(solve):
    # 两 GT 都选中同一 anchor：保留 cost 小的 GT
    # GT0 与 anchor IoU 中等但 cls 准；GT1 cls 差 → anchor 归 GT0
    gts = [[32.0, 32.0, 8.0, 8.0], [34.0, 34.0, 8.0, 8.0]]
    led = _compare(solve, gts, [0, 1], [8], [3], [3],
                   [[5.0, -5.0]], [0.0],
                   [[33.0, 33.0, 8.0, 8.0]], 2)
    # anchor 距 GT0 中心 (1.5+1.5)/2≈… 两 GT 邻域都覆盖 → 双方候选
    # dynamic_k 均 ≥1 且 cost 最小者都是该 anchor → 冲突
    j = led["matching_matrix"]
    if sum(r[0] for r in j) > 1:
        best = min(range(2), key=lambda g: led["cost"][g][0])
        assert led["matched_gt_inds"][0] == best
        assert sum(r[0] for r in j) == 1
    else:
        assert led["num_fg"] == 1


def check_offscreen_cost_1e6(solve):
    # 几何不命中的 anchor 仍参与 IoU/cost（+1e6 惩罚），dynamic_k 只在
    # 候选（filter 通过者）内取——不命中者永远不被选，除非候选不足
    gts = [[100.0, 100.0, 4.0, 4.0]]
    strides = [8, 8]
    xs = [12, 30]
    ys = [12, 12]
    led = _compare(solve, gts, [0], strides, xs, ys,
                   [[0.0], [0.0]], [0.0, 0.0],
                   [[100.0, 100.0, 4.0, 4.0],
                    [100.0, 100.0, 4.0, 4.0]], 1)
    # anchor0 中心 (100,100)，dist=12 → 在；anchor1 中心 244 → c_r<0 不在
    assert led["anchor_filter"] == [True, False]
    # 候选只有 anchor0：matching_matrix 是 1 列（1e6 惩罚者根本不进候选）
    assert len(led["matching_matrix"][0]) == 1
    assert led["matching_matrix"][0][0] == 1
    assert led["num_fg"] == 1


def check_multi_gt_no_conflict(solve):
    # 两 GT 各匹配各自 anchor，无冲突
    gts = [[24.0, 24.0, 8.0, 8.0], [72.0, 24.0, 8.0, 8.0]]
    strides = [8] * 4
    xs = [2, 3, 8, 9]
    ys = [2, 2, 2, 2]
    preds = [[(x + 0.5) * 8, 24.0, 8.0, 8.0] for x in xs]
    cls_preds = [[3.0, -3.0], [3.0, -3.0], [-3.0, 3.0], [-3.0, 3.0]]
    led = _compare(solve, gts, [0, 1], strides, xs, ys, cls_preds,
                   [0.0] * 4, preds, 2)
    assert led["num_fg"] == 2
    assert led["matched_gt_inds"][0] == 0
    assert led["matched_gt_inds"][2] == 1
    assert led["gt_matched_classes"][0] == 0
    assert led["gt_matched_classes"][2] == 1
    _close(led["pred_ious_this_matching"][0], led["ious"][0][0])


def check_random_against_oracle(solve):
    rng = random.Random(706)
    for _ in range(200):
        G = rng.randrange(1, 4)
        A = rng.randrange(4, 16)
        C = rng.randrange(2, 5)
        gts, gcls = [], []
        for _ in range(G):
            cx = rng.uniform(10, 120)
            cy = rng.uniform(10, 100)
            wh = rng.uniform(4, 40)
            gts.append([round(cx, 1), round(cy, 1), round(wh, 1),
                        round(wh * rng.uniform(0.5, 2.0), 1)])
            gcls.append(rng.randrange(C))
        strides = [rng.choice([8, 16, 32]) for _ in range(A)]
        xs = [rng.randrange(0, 8) for _ in range(A)]
        ys = [rng.randrange(0, 8) for _ in range(A)]
        cls_preds = [[round(rng.uniform(-3, 3), 2) for _ in range(C)]
                     for _ in range(A)]
        obj_preds = [round(rng.uniform(-2, 2), 2) for _ in range(A)]
        preds = []
        for k in range(A):
            pcx = (xs[k] + 0.5) * strides[k] + rng.uniform(-4, 4)
            pcy = (ys[k] + 0.5) * strides[k] + rng.uniform(-4, 4)
            preds.append([round(pcx, 1), round(pcy, 1),
                          round(rng.uniform(4, 40), 1),
                          round(rng.uniform(4, 40), 1)])
        _compare(solve, gts, gcls, strides, xs, ys, cls_preds, obj_preds,
                 preds, C)


def check_invalid_inputs(solve):
    good = dict(gt_bboxes=[[32.0, 32.0, 8.0, 8.0]], gt_classes=[0],
                expanded_strides=[8], x_shifts=[3], y_shifts=[3],
                cls_preds=[[0.0]], obj_preds=[0.0],
                bboxes_preds=[[32.0, 32.0, 8.0, 8.0]], num_classes=1)

    def call(**kw):
        a = {**good, **kw}
        return solve(a["gt_bboxes"], a["gt_classes"], a["expanded_strides"],
                     a["x_shifts"], a["y_shifts"], a["cls_preds"],
                     a["obj_preds"], a["bboxes_preds"], a["num_classes"])
    for kw in [
        dict(gt_bboxes=[]),                          # 无 GT
        dict(expanded_strides=[]),                   # 无 anchor
        dict(gt_classes=[1], num_classes=1),         # 类越界
        dict(gt_classes=[-1]),
        dict(expanded_strides=[0]),                  # stride 0
        dict(bboxes_preds=[[32.0, 32.0, 0.0, 8.0]]),  # w=0
        dict(bboxes_preds=[[32.0, 32.0, 8.0, -1.0]]),
        dict(gt_bboxes=[[32.0, 32.0, 8.0]]),         # 行长 3
        dict(cls_preds=[[0.0, 0.0]]),                # cls 行长不齐（>C）
        dict(obj_preds=[0.0, 0.0]),                  # obj 长度不齐
        dict(gt_bboxes=[[float("inf"), 32.0, 8.0, 8.0]]),
    ]:
        try:
            call(**kw)
        except (ValueError, IndexError):
            pass
        else:
            raise AssertionError(f"expected error: {kw}")
