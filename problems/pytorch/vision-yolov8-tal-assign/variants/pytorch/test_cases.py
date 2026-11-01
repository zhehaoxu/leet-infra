"""YOLOv8 TAL 分配账（test cases）。

独立 oracle：显式枚举结构（无 max/next 抽象）+ 200 组随机对拍。
"""

import math
import random


def _ciou(gt, pd):
    ix1, iy1 = max(gt[0], pd[0]), max(gt[1], pd[1])
    ix2, iy2 = min(gt[2], pd[2]), min(gt[3], pd[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ug = (gt[2] - gt[0]) * (gt[3] - gt[1])
    up = (pd[2] - pd[0]) * (pd[3] - pd[1])
    un = ug + up - inter
    iou = inter / un if un > 0 else 0.0
    rho2 = ((gt[0] + gt[2]) / 2 - (pd[0] + pd[2]) / 2) ** 2 + \
        ((gt[1] + gt[3]) / 2 - (pd[1] + pd[3]) / 2) ** 2
    c2 = (max(gt[2], pd[2]) - min(gt[0], pd[0])) ** 2 + \
        (max(gt[3], pd[3]) - min(gt[1], pd[1])) ** 2
    v = (4 / math.pi ** 2) * (math.atan((pd[2] - pd[0]) / (pd[3] - pd[1]))
                              - math.atan((gt[2] - gt[0]) / (gt[3] - gt[1]))) ** 2
    den = 1 - iou + v
    a = v / den if den > 0 else 0.0
    return max(0.0, iou - rho2 / c2 - a * v)


def _oracle(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes,
            num_classes, topk, alpha, beta, stride_val, eps):
    G, A = len(gt_bboxes), len(pd_bboxes)
    # 扩边
    exp = []
    for g in range(G):
        x1, y1, x2, y2 = gt_bboxes[g]
        w, h = x2 - x1, y2 - y1
        if w < stride_val:
            c = (x1 + x2) / 2
            x1, x2 = c - stride_val / 2, c + stride_val / 2
        if h < stride_val:
            c = (y1 + y2) / 2
            y1, y2 = c - stride_val / 2, c + stride_val / 2
        exp.append([x1, y1, x2, y2])
    in_gts = [[anc_points[k][0] - exp[g][0] > eps
               and anc_points[k][1] - exp[g][1] > eps
               and exp[g][2] - anc_points[k][0] > eps
               and exp[g][3] - anc_points[k][1] > eps
               for k in range(A)] for g in range(G)]
    scores, overs, align = [], [], []
    for g in range(G):
        sr, orr, ar = [], [], []
        for k in range(A):
            if in_gts[g][k]:
                s = pd_scores[k][gt_labels[g]]
                o = _ciou(gt_bboxes[g], pd_bboxes[k])
            else:
                s, o = 0.0, 0.0
            sr.append(s)
            orr.append(o)
            ar.append(s ** alpha * o ** beta if in_gts[g][k] else 0.0)
        scores.append(sr)
        overs.append(orr)
        align.append(ar)
    topk_m = [[0] * A for _ in range(G)]
    for g in range(G):
        cand = sorted(range(A), key=lambda k: (-align[g][k], k))[:topk]
        for k in cand:
            topk_m[g][k] = 1
    pos = [[topk_m[g][k] == 1 and in_gts[g][k] for k in range(A)]
           for g in range(G)]
    # 仲裁
    pos_pre = [row[:] for row in pos]
    for k in range(A):
        n = sum(pos[g][k] for g in range(G))
        if n > 1:
            bg = 0
            for g in range(1, G):
                if pos[g][k] and (overs[g][k] > overs[bg][k]
                                  or (overs[g][k] == overs[bg][k]
                                      and not pos[bg][k])):
                    bg = g
            for g in range(G):
                pos[g][k] = False
            pos[bg][k] = True
    fg = [any(pos[g][k] for g in range(G)) for k in range(A)]
    tgi = []
    for k in range(A):
        if fg[k]:
            g = 0
            while not pos[g][k]:
                g += 1
            tgi.append(g)
        else:
            tgi.append(-1)
    pa = []
    po = []
    for g in range(G):
        vals_a = [align[g][k] for k in range(A) if pos[g][k]]
        vals_o = [overs[g][k] for k in range(A) if pos[g][k]]
        pa.append(max(vals_a) if vals_a else 0.0)
        po.append(max(vals_o) if vals_o else 0.0)
    norm = []
    for k in range(A):
        best = 0.0
        for g in range(G):
            if pos[g][k]:
                v = align[g][k] * po[g] / (pa[g] + eps)
                if v > best:
                    best = v
        norm.append(best)
    ts = []
    for k in range(A):
        row = [0.0] * num_classes
        if fg[k]:
            row[gt_labels[tgi[k]]] = norm[k]
        ts.append(row)
    return {"expanded_boxes": exp, "mask_in_gts": in_gts,
            "bbox_scores": scores, "overlaps": overs, "align_metric": align,
            "mask_topk": topk_m,
            "mask_pos": [[int(v) for v in r] for r in pos_pre],
            "mask_pos_arb": [[int(v) for v in r] for r in pos],
            "fg_mask": fg, "target_gt_idx": tgi,
            "pos_align_metrics": pa, "pos_overlaps": po,
            "norm_align_metric": norm, "target_scores": ts}


def _close(a, b, tol=1e-9):
    assert math.isclose(a, b, rel_tol=tol, abs_tol=tol), f"{a} != {b}"


def _compare(solve, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes,
             num_classes, topk=13, alpha=1.0, beta=6.0, stride_val=16.0,
             eps=1e-9):
    led = solve(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes,
                num_classes, topk, alpha, beta, stride_val, eps)
    ref = _oracle(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes,
                  num_classes, topk, alpha, beta, stride_val, eps)
    G, A = len(gt_bboxes), len(pd_bboxes)
    for g in range(G):
        for a, b in zip(led["expanded_boxes"][g], ref["expanded_boxes"][g]):
            _close(a, b)
        assert led["mask_in_gts"][g] == ref["mask_in_gts"][g]
        for k in range(A):
            _close(led["bbox_scores"][g][k], ref["bbox_scores"][g][k])
            _close(led["overlaps"][g][k], ref["overlaps"][g][k])
            _close(led["align_metric"][g][k], ref["align_metric"][g][k])
        assert led["mask_topk"][g] == ref["mask_topk"][g]
    assert led["mask_pos"] == ref["mask_pos"]
    assert led["mask_pos_arb"] == ref["mask_pos_arb"]
    assert led["fg_mask"] == ref["fg_mask"]
    assert led["target_gt_idx"] == ref["target_gt_idx"]
    for g in range(G):
        _close(led["pos_align_metrics"][g], ref["pos_align_metrics"][g])
        _close(led["pos_overlaps"][g], ref["pos_overlaps"][g])
    for k in range(A):
        _close(led["norm_align_metric"][k], ref["norm_align_metric"][k])
        for c in range(num_classes):
            _close(led["target_scores"][k][c], ref["target_scores"][k][c])
    return led


def _grid_anchors(W, H, stride=1.0, offset=0.5):
    return [[(x + offset) * stride, (y + offset) * stride]
            for y in range(H) for x in range(W)]


def check_single_gt_center_pick(solve):
    # 单 GT：中心在框内的 anchor 候选；align 由分数与 CIoU 共同决定
    G = 1
    gt = [[10.0, 10.0, 40.0, 40.0]]
    anc = _grid_anchors(8, 8, stride=5.0)          # 64 个候选 ≥ topk
    pd_scores = [[0.9 if k == 27 else 0.01] for k in range(64)]
    # 预测框 27 号恰与 GT 重合
    pd_bboxes = []
    for k in range(64):
        ax, ay = anc[k]
        if k == 27:
            pd_bboxes.append([10.0, 10.0, 40.0, 40.0])
        else:
            pd_bboxes.append([ax - 2, ay - 2, ax + 2, ay + 2])
    led = _compare(solve, pd_scores, pd_bboxes, anc, [0], gt, 1, topk=4)
    assert led["mask_in_gts"][0][27] is True
    # 27 号 CIoU=1（重合），align = 0.9·1 = 0.9 最大 → topk 首位
    _close(led["overlaps"][0][27], 1.0)
    _close(led["align_metric"][0][27], 0.9)
    assert led["mask_topk"][0][27] == 1
    assert led["target_gt_idx"][27] == 0
    # norm：pos_overlap=1、pos_align=0.9 → norm(27) = 0.9·1/0.9 = 1.0
    _close(led["norm_align_metric"][27], 1.0, 1e-7)
    _close(led["target_scores"][27][0], 1.0, 1e-7)


def check_center_strict_margin(solve):
    # 中心恰在框边（delta=0）不算（>eps 严格）；框内即候选
    gt = [[0.0, 0.0, 20.0, 20.0]]
    anc = [[0.0, 10.0], [10.0, 10.0], [20.0, 10.0], [10.0, 0.0],
           [5.0, 5.0]] + [[100.0 + k, 100.0] for k in range(15)]
    pd_scores = [[0.5] for _ in range(20)]
    pd_bboxes = [[5.0, 5.0, 15.0, 15.0]] * 20
    led = _compare(solve, pd_scores, pd_bboxes, anc, [0], gt, 1, topk=3)
    assert led["mask_in_gts"][0][:5] == [False, True, False, False, True]


def check_eps_margin_exact(solve):
    # anchor 距框边恰 = eps（x1=0 使浮点差恰等）→ 官方 >eps 不算在内
    gt = [[0.0, 0.0, 20.0, 20.0]]
    anc = [[1e-9, 10.0], [5.0, 10.0]] + \
        [[100.0 + k, 100.0] for k in range(11)]
    pd_scores = [[0.5] for _ in range(13)]
    pd_bboxes = [[5.0, 5.0, 15.0, 15.0]] * 13
    led = _compare(solve, pd_scores, pd_bboxes, anc, [0], gt, 1, topk=3)
    assert led["mask_in_gts"][0][0] is False   # delta == eps 恰等 → False
    assert led["mask_in_gts"][0][1] is True


def check_small_side_enlargement(solve):
    # 小于 stride_val 的边按中心扩到 stride_val → 更多中心候选
    gt = [[10.0, 10.0, 12.0, 60.0]]              # w=2 < 16 → 扩到 16
    anc = [[5.9, 30.0], [18.1, 30.0], [10.0, 30.0], [1.0, 30.0],
           [21.0, 30.0]] + [[100.0 + k, 100.0] for k in range(15)]
    pd_scores = [[0.5] for _ in range(20)]
    pd_bboxes = [[10.0, 10.0, 12.0, 60.0]] * 20
    led = _compare(solve, pd_scores, pd_bboxes, anc, [0], gt, 1, topk=3)
    # 扩边后 x ∈ [3, 19]（中心 11 对称扩到 16 宽）：5.9/18.1/10 在内
    assert led["mask_in_gts"][0][:5] == [True, True, True, False, False]
    _close(led["expanded_boxes"][0][0], 3.0)
    _close(led["expanded_boxes"][0][2], 19.0)
    _close(led["expanded_boxes"][0][1], 10.0)     # h=50 ≥16 不扩
    _close(led["expanded_boxes"][0][3], 60.0)


def check_ciou_clamp_negative(solve):
    # CIoU 为负（预测框偏远）→ clamp 0 → align=0
    gt = [[10.0, 10.0, 40.0, 40.0]]
    anc = _grid_anchors(8, 8, stride=5.0)
    pd_scores = [[0.8] for _ in range(64)]
    pd_bboxes = []
    for k in range(64):
        ax, ay = anc[k]
        pd_bboxes.append([ax - 2, ay - 2, ax + 2, ay + 2])
    led = _compare(solve, pd_scores, pd_bboxes, anc, [0], gt, 1, topk=4)
    # 框内 anchor 的预测框可能完全偏离 → CIoU<0 → 0
    for k in range(64):
        if led["mask_in_gts"][0][k]:
            assert led["overlaps"][0][k] >= 0.0
    # align 全在 [0, 0.8]
    for v in led["align_metric"][0]:
        assert 0.0 <= v <= 0.8 + 1e-12


def check_alpha_beta_exponent(solve):
    # alpha/beta 幂次：alpha=2、beta=1 下 align = s²·iou
    gt = [[10.0, 10.0, 40.0, 40.0]]
    anc = _grid_anchors(8, 8, stride=5.0)
    pd_scores = [[0.5] for _ in range(64)]
    pd_bboxes = [[10.0, 10.0, 40.0, 40.0]] * 64
    led = _compare(solve, pd_scores, pd_bboxes, anc, [0], gt, 1,
                   topk=4, alpha=2.0, beta=1.0)
    k0 = next(k for k in range(64) if led["mask_in_gts"][0][k])
    _close(led["align_metric"][0][k0], 0.5 ** 2 * led["overlaps"][0][k0])


def check_multi_gt_arbitration_by_overlap(solve):
    # 两 GT 争同一 anchor：留 overlap 大者（非 align 大者）
    gt = [[10.0, 10.0, 30.0, 30.0], [12.0, 12.0, 32.0, 32.0]]
    anc = [[20.0, 20.0]] + [[100.0 + k, 100.0 + k] for k in range(13)]
    pd_scores = [[0.9, 0.9]] * 14
    # 预测框偏向 GT1 → 与 GT1 的 CIoU 更大
    pd_bboxes = [[13.0, 13.0, 31.0, 31.0]] + \
        [[90.0 + k, 90.0 + k, 95.0 + k, 95.0 + k] for k in range(13)]
    led = _compare(solve, pd_scores, pd_bboxes, anc, [0, 1], gt, 2, topk=13)
    # anchor0 在两 GT 框内 → 仲裁；pred 偏 GT1 → overlap(GT1) > overlap(GT0)
    assert led["mask_pos"][0][0] == 1 and led["mask_pos"][1][0] == 1
    assert led["mask_pos_arb"][0][0] == 0 and led["mask_pos_arb"][1][0] == 1
    assert led["target_gt_idx"][0] == 1


def check_tie_arbitration_smaller_gt(solve):
    # overlap 平分时取小 GT 下标：两 GT 与 pred 完全重合
    gt = [[10.0, 10.0, 30.0, 30.0], [10.0, 10.0, 30.0, 30.0]]
    anc = [[20.0, 20.0]] + [[100.0 + k, 100.0 + k] for k in range(13)]
    pd_scores = [[0.9, 0.9]] * 14
    pd_bboxes = [[10.0, 10.0, 30.0, 30.0]] + \
        [[90.0 + k, 90.0 + k, 95.0 + k, 95.0 + k] for k in range(13)]
    led = _compare(solve, pd_scores, pd_bboxes, anc, [0, 1], gt, 2, topk=13)
    assert led["target_gt_idx"][0] == 0
    assert led["mask_pos_arb"][1][0] == 0


def check_soft_label_normalization(solve):
    # 软标签：target_scores[gt 类位] = align·pos_overlap/(pos_align+eps)
    # 归一后最优 anchor 的软标签 = pos_overlap（其 align==pos_align）
    gt = [[10.0, 10.0, 40.0, 40.0]]
    anc = _grid_anchors(8, 8, stride=5.0)
    pd_scores = [[0.6 if k % 2 == 0 else 0.3] for k in range(64)]
    pd_bboxes = []
    for k in range(64):
        ax, ay = anc[k]
        if k == 0:
            pd_bboxes.append([10.0, 10.0, 40.0, 40.0])
        else:
            pd_bboxes.append([ax - 3, ay - 3, ax + 3, ay + 3])
    led = _compare(solve, pd_scores, pd_bboxes, anc, [0], gt, 1, topk=5)
    k_best = max((k for k in range(64) if led["mask_pos"][0][k]),
                 key=lambda k: led["align_metric"][0][k])
    # norm = align·pos_overlap/(pos_align+eps)；pos_align >> eps 时
    # 最优 anchor 的软标签收敛到 pos_overlap（align==pos_align）
    _close(led["norm_align_metric"][k_best],
           led["align_metric"][0][k_best] * led["pos_overlaps"][0]
           / (led["pos_align_metrics"][0] + 1e-9))
    # 大 align 用例：pos_align=0.9 >> eps → norm(27)=pos_overlap=1
    anc2 = _grid_anchors(8, 8, stride=5.0)
    ps2 = [[0.9 if k == 27 else 0.01] for k in range(64)]
    pb2 = []
    for k in range(64):
        ax, ay = anc2[k]
        pb2.append([10.0, 10.0, 40.0, 40.0] if k == 27
                   else [ax - 2, ay - 2, ax + 2, ay + 2])
    led2 = _compare(solve, ps2, pb2, anc2, [0], gt, 1, topk=4)
    _close(led2["norm_align_metric"][27], led2["pos_overlaps"][0], 1e-7)
    _close(led2["target_scores"][27][0], led2["pos_overlaps"][0], 1e-7)


def check_random_against_oracle(solve):
    rng = random.Random(708)
    for _ in range(200):
        G = rng.randrange(1, 4)
        W, H = 6, 6
        stride = rng.choice([4.0, 8.0])
        anc = _grid_anchors(W, H, stride=stride)
        A = len(anc)
        topk = rng.choice([3, 5, 8])
        if A < topk:
            topk = A
        C = rng.randrange(1, 4)
        pd_scores = [[round(rng.uniform(0.01, 0.99), 3) for _ in range(C)]
                     for _ in range(A)]
        pd_bboxes = []
        for k in range(A):
            ax, ay = anc[k]
            s = rng.uniform(2, 12)
            pd_bboxes.append([round(ax - s / 2, 1), round(ay - s / 2, 1),
                              round(ax + s / 2, 1), round(ay + s / 2, 1)])
        gts, gcls = [], []
        for _ in range(G):
            cx = rng.uniform(8, W * stride - 8)
            cy = rng.uniform(8, H * stride - 8)
            s = rng.uniform(4, 30)
            gts.append([round(cx - s / 2, 1), round(cy - s / 2, 1),
                        round(cx + s / 2, 1), round(cy + s / 2, 1)])
            gcls.append(rng.randrange(C))
        sv = rng.choice([8.0, 16.0])
        _compare(solve, pd_scores, pd_bboxes, anc, gcls, gts, C,
                 topk=topk, stride_val=sv)


def check_invalid_inputs(solve):
    good = dict(pd_scores=[[0.5]] * 13,
                pd_bboxes=[[0.0, 0.0, 10.0, 10.0]] * 13,
                anc_points=[[5.0, 5.0]] * 13,
                gt_labels=[0], gt_bboxes=[[2.0, 2.0, 8.0, 8.0]],
                num_classes=1)

    def call(**kw):
        a = {**good, **kw}
        return solve(a["pd_scores"], a["pd_bboxes"], a["anc_points"],
                     a["gt_labels"], a["gt_bboxes"], a["num_classes"],
                     a.get("topk", 13))
    for kw in [
        dict(gt_labels=[], gt_bboxes=[]),            # 无 GT
        dict(pd_scores=[], pd_bboxes=[], anc_points=[]),  # 无 anchor
        dict(topk=14),                               # A < topk
        dict(topk=0),
        dict(pd_scores=[[0.5, 0.5]] * 13),           # 行长 ≠ C
        dict(pd_scores=[[1.5]] * 13),                # 概率 >1
        dict(pd_scores=[[-0.1]] * 13),
        dict(gt_labels=[1], num_classes=1),          # 类越界
        dict(gt_bboxes=[[8.0, 2.0, 2.0, 8.0]]),      # x1 > x2
        dict(pd_bboxes=[[0.0, 0.0, 0.0, 10.0]] * 13),  # 退化 w
        dict(anc_points=[[5.0]] * 13),               # 点一维
        dict(gt_bboxes=[[float("nan"), 2.0, 8.0, 8.0]]),
    ]:
        try:
            call(**kw)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError(f"expected error: {kw}")
