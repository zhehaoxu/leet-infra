import math

import torch

# 示例（实测回填）：A 为轴对齐 2x2，B 是同一框绕中心逆时针转 45°，C 在远处
EXAMPLE_THETA = math.pi / 4
EXAMPLE_BOXES = [[0.0, 0.0, 2.0, 2.0, 0.0],
                 [0.0, 0.0, 2.0, 2.0, EXAMPLE_THETA],
                 [10.0, 10.0, 2.0, 2.0, 0.0]]
EXAMPLE_SCORES = [0.9, 0.8, 0.7]
EXAMPLE_IOU_01 = 0.7071067811865476  # 实测，理论上等于 1/sqrt(2)
EXAMPLE_KEEP = [0, 2]
EXAMPLE_MASK = [True, False, True]


def _boxes(rows, dtype=torch.float64):
    return torch.tensor(rows, dtype=dtype)


def _scores(vals, dtype=torch.float64):
    return torch.tensor(vals, dtype=dtype)


def check_example(solve):
    keep, mask, iou = solve(_boxes(EXAMPLE_BOXES), _scores(EXAMPLE_SCORES), iou_thr=0.5)
    assert iou.shape == (3, 3) and iou.dtype == torch.float64, (iou.shape, iou.dtype)
    assert keep.dtype == torch.int64 and keep.shape == (2,)
    assert mask.dtype == torch.bool and mask.shape == (3,)
    assert abs(iou[0, 1].item() - EXAMPLE_IOU_01) < 1e-12, iou[0, 1].item()
    assert abs(iou[0, 0].item() - 1.0) < 1e-12 and abs(iou[2, 2].item() - 1.0) < 1e-12
    assert iou[0, 2].item() == 0.0 and iou[1, 2].item() == 0.0
    assert keep.tolist() == EXAMPLE_KEEP, keep.tolist()
    assert mask.tolist() == EXAMPLE_MASK, mask.tolist()


check_example.example = True


def _corners(box):
    cx, cy, w, h, th = [float(v) for v in box]
    c, s = math.cos(th), math.sin(th)
    out = []
    for ox, oy in ((w / 2.0, -h / 2.0), (w / 2.0, h / 2.0),
                   (-w / 2.0, h / 2.0), (-w / 2.0, -h / 2.0)):
        out.append((ox * c - oy * s + cx, ox * s + oy * c + cy))
    return out


def _poly_area(pts):
    if len(pts) < 3:
        return 0.0
    total = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def _clip(subject, clip):
    """纯 Python 裁剪：逐边按 (p -> q) 的出点顺序收集，与参考解的下标方向相反。"""
    out = list(subject)
    for i in range(len(clip)):
        ax, ay = clip[i]
        bx, by = clip[(i + 1) % len(clip)]
        ex, ey = bx - ax, by - ay
        inp, out = out, []
        if not inp:
            return []
        for k in range(len(inp)):
            px, py = inp[k]
            qx, qy = inp[(k + 1) % len(inp)]
            sp = ex * (py - ay) - ey * (px - ax)
            sq = ex * (qy - ay) - ey * (qx - ax)
            if sq >= 0.0:
                if sp < 0.0:
                    t = sp / (sp - sq)
                    out.append((px + t * (qx - px), py + t * (qy - py)))
                out.append((qx, qy))
            elif sp >= 0.0:
                t = sp / (sp - sq)
                out.append((px + t * (qx - px), py + t * (qy - py)))
    return out


def _iou_pair(a, b, pa, pb):
    aa = _poly_area(pa)
    ab = _poly_area(pb)
    inter = _poly_area(_clip(pa, pb))
    inter = max(0.0, min(inter, aa, ab))
    union = aa + ab - inter
    return inter / union if union > 0.0 else 0.0


def _oracle(boxes, scores, thr):
    rows = boxes.tolist()
    sc = [float(v) for v in scores.tolist()]
    n = len(rows)
    polys = [_corners(r) for r in rows]
    mat = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            v = _iou_pair(rows[i], rows[j], polys[i], polys[j])
            mat[i][j] = mat[j][i] = v
    order = sorted(range(n), key=lambda i: (-sc[i], i))
    kept = []
    for idx in order:
        if all(mat[idx][k] <= thr for k in kept):
            kept.append(idx)
    mask = [False] * n
    for k in kept:
        mask[k] = True
    return mat, kept, mask


def check_oracle(solve):
    """任意角度、含大角度与近重叠框的配置，与纯 Python 裁剪 oracle 全量对拍。"""
    torch.manual_seed(17)
    for ci in range(3):
        n = 6 - ci
        xy = torch.randn(n, 2) * 3.0
        wh = torch.rand(n, 2) * 4.0 + 0.5
        th = torch.rand(n) * 3.0 - 1.5
        if ci == 2:  # 制造高度重叠：同一中心的多个旋转副本
            xy = torch.zeros(n, 2)
            wh = torch.full((n, 2), 2.0)
            th = torch.arange(n, dtype=torch.float64) * (math.pi / n)
        boxes = torch.cat([xy, wh, th.unsqueeze(1)], dim=1)
        scores = torch.rand(n, dtype=boxes.dtype)
        for thr in (0.3, 0.5):
            keep, mask, mat = solve(boxes, scores, iou_thr=thr)
            exp_mat, exp_keep, exp_mask = _oracle(boxes, scores, thr)
            exp_t = torch.tensor(exp_mat, dtype=torch.float64)
            assert torch.allclose(mat, exp_t, atol=1e-9), \
                (ci, thr, (mat - exp_t).abs().max().item())
            assert keep.tolist() == exp_keep, (ci, thr, keep.tolist(), exp_keep)
            assert mask.tolist() == exp_mask, (ci, thr, mask.tolist(), exp_mask)


def _rect_iou(a, b):
    """轴对齐矩形的解析 IoU，(cx,cy,w,h)。"""
    ax1, ax2 = a[0] - a[2] / 2.0, a[0] + a[2] / 2.0
    ay1, ay2 = a[1] - a[3] / 2.0, a[1] + a[3] / 2.0
    bx1, bx2 = b[0] - b[2] / 2.0, b[0] + b[2] / 2.0
    by1, by2 = b[1] - b[3] / 2.0, b[1] + b[3] / 2.0
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0.0 else 0.0


def check_axis_aligned(solve):
    """theta=0 时旋转 IoU 必须等于轴对齐矩形 IoU（解析公式，独立路径）。"""
    torch.manual_seed(23)
    for _ in range(20):
        a = torch.cat([torch.randn(2) * 2.0, torch.rand(2) * 5.0 + 0.5])
        b = torch.cat([torch.randn(2) * 2.0, torch.rand(2) * 5.0 + 0.5])
        boxes = torch.stack([torch.cat([a, torch.zeros(1)]),
                             torch.cat([b, torch.zeros(1)])]).to(torch.float64)
        mat = solve(boxes, torch.tensor([0.9, 0.8], dtype=torch.float64))[2]
        exp = _rect_iou(a.tolist(), b.tolist())
        assert abs(mat[0, 1].item() - exp) < 1e-12, (mat[0, 1].item(), exp)


def check_geometry_analytic(solve):
    """45°、90°、包含、同一框、互斥的解析值。"""
    # 同一正方形转 45°：交集 2(sqrt2-1)s^2，IoU = 1/sqrt2
    boxes = _boxes([[0, 0, 2, 2, 0], [0, 0, 2, 2, EXAMPLE_THETA]])
    mat = solve(boxes, _scores([0.9, 0.8]))[2]
    assert abs(mat[0, 1].item() - 1.0 / math.sqrt(2)) < 1e-9, mat[0, 1].item()
    # 4x2 的框转 90°：交集 2x2=4，并集 8+8-4=12 -> 1/3
    boxes = _boxes([[1, 1, 4, 2, 0], [1, 1, 4, 2, math.pi / 2]])
    mat = solve(boxes, _scores([0.9, 0.8]))[2]
    assert abs(mat[0, 1].item() - 1.0 / 3.0) < 1e-12, mat[0, 1].item()
    # 包含：2x2 在 10x10 内 -> 4/100 = 0.04
    boxes = _boxes([[5, 5, 10, 10, 0], [5, 5, 2, 2, 0]])
    assert abs(solve(boxes, _scores([0.9, 0.8]))[2][0, 1].item() - 0.04) < 1e-12
    # 同一个框（含旋转）-> 1.0
    boxes = _boxes([[1, 2, 3, 4, 0.37], [1, 2, 3, 4, 0.37]])
    assert abs(solve(boxes, _scores([0.9, 0.8]))[2][0, 1].item() - 1.0) < 1e-12
    # 互斥 -> 0
    boxes = _boxes([[0, 0, 2, 2, 0.5], [20, 20, 2, 2, 0.5]])
    assert solve(boxes, _scores([0.9, 0.8]))[2][0, 1].item() == 0.0


def check_rotation_invariance(solve):
    """两个框同时旋转同一个角度，IoU 不变（几何不变量）。"""
    torch.manual_seed(31)
    phi = 0.37
    for _ in range(8):
        a = torch.cat([torch.randn(2) * 2.0, torch.rand(2) * 4.0 + 0.5,
                       torch.rand(1) * 2.0 - 1.0])
        b = torch.cat([torch.randn(2) * 2.0, torch.rand(2) * 4.0 + 0.5,
                       torch.rand(1) * 2.0 - 1.0])
        boxes = torch.stack([a, b]).to(torch.float64)
        rot = boxes.clone()
        c, s = math.cos(phi), math.sin(phi)
        rot[:, 0] = boxes[:, 0] * c - boxes[:, 1] * s
        rot[:, 1] = boxes[:, 0] * s + boxes[:, 1] * c
        rot[:, 4] = boxes[:, 4] + phi
        v0 = solve(boxes, _scores([0.9, 0.8]))[2][0, 1].item()
        v1 = solve(rot, _scores([0.9, 0.8]))[2][0, 1].item()
        assert abs(v0 - v1) < 1e-9, (v0, v1)


def check_nms_rules(solve):
    """贪心顺序、同分取小下标、阈值严格性、与所有已保留框比较。"""
    # 同分：[0.5, 0.9, 0.9, 0.7] -> 处理顺序 1,2,3,0；框 2 与框 0 完全重合
    boxes = _boxes([[0, 0, 2, 2, 0], [50, 50, 2, 2, 0], [0, 0, 2, 2, 0], [51, 51, 2, 2, 0]])
    scores = _scores([0.5, 0.9, 0.9, 0.7])
    keep, mask, mat = solve(boxes, scores, iou_thr=0.5)
    assert keep.tolist() == [1, 2, 3], keep.tolist()
    assert mask.tolist() == [False, True, True, True], mask.tolist()
    assert sorted(keep.tolist()) == [i for i, v in enumerate(mask.tolist()) if v]
    # 链式：相邻 1/3、首尾不相交；只与上一个比会把 2 错误抑制
    boxes = _boxes([[5, 5, 10, 10, 0], [10, 5, 10, 10, 0], [15, 5, 10, 10, 0]])
    scores = _scores([0.9, 0.8, 0.7])
    keep, mask, mat = solve(boxes, scores, iou_thr=0.3)
    assert abs(mat[0, 1].item() - 1.0 / 3.0) < 1e-12
    assert abs(mat[1, 2].item() - 1.0 / 3.0) < 1e-12 and mat[0, 2].item() == 0.0
    assert keep.tolist() == [0, 2], keep.tolist()
    # 阈值严格性：IoU 恰为 1/3 时 iou_thr=1/3 保留、0.3 抑制
    boxes = _boxes([[5, 5, 10, 10, 0], [10, 5, 10, 10, 0]])
    scores = _scores([0.9, 0.8])
    assert solve(boxes, scores, iou_thr=1.0 / 3.0)[0].tolist() == [0, 1]
    assert solve(boxes, scores, iou_thr=0.3)[0].tolist() == [0]
    # 完全重合（IoU=1）时 iou_thr=1.0 保留两个、0.99 只留一个
    boxes = _boxes([[0, 0, 10, 10, 0.3], [0, 0, 10, 10, 0.3]])
    scores = _scores([0.9, 0.8])
    assert abs(solve(boxes, scores)[2][0, 1].item() - 1.0) < 1e-12
    assert solve(boxes, scores, iou_thr=1.0)[0].tolist() == [0, 1]
    assert solve(boxes, scores, iou_thr=0.99)[0].tolist() == [0]
    # 互斥框在 iou_thr=0 下都保留（0 > 0 为假）
    boxes = _boxes([[0, 0, 2, 2, 0.5], [20, 20, 2, 2, 0.5]])
    assert solve(boxes, _scores([0.9, 0.8]), iou_thr=0.0)[0].tolist() == [0, 1]


def check_matrix_properties(solve):
    torch.manual_seed(41)
    n = 8
    boxes = torch.cat([torch.randn(n, 2) * 4.0, torch.rand(n, 2) * 4.0 + 0.5,
                       (torch.rand(n, 1) * 2.0 - 1.0)], dim=1).to(torch.float64)
    keep, mask, mat = solve(boxes, torch.rand(n, dtype=torch.float64))
    assert mat.shape == (n, n) and mat.dtype == torch.float64
    assert torch.allclose(mat, mat.t(), atol=1e-12), (mat - mat.t()).abs().max().item()
    assert torch.allclose(torch.diagonal(mat), torch.ones(n, dtype=torch.float64))
    assert bool((mat >= 0.0).all()) and bool((mat <= 1.0 + 1e-12).all())
    assert len(set(keep.tolist())) == keep.numel()
    assert sorted(keep.tolist()) == [i for i, v in enumerate(mask.tolist()) if v]


def check_empty(solve):
    keep, mask, mat = solve(torch.zeros(0, 5), torch.zeros(0))
    assert keep.shape == (0,) and keep.dtype == torch.int64
    assert mask.shape == (0,) and mask.dtype == torch.bool
    assert mat.shape == (0, 0) and mat.dtype == torch.float64


def check_dtype(solve):
    keep64, mask64, mat64 = solve(_boxes(EXAMPLE_BOXES), _scores(EXAMPLE_SCORES))
    keep32, mask32, mat32 = solve(_boxes(EXAMPLE_BOXES, torch.float32),
                                 _scores(EXAMPLE_SCORES, torch.float32))
    assert mat32.dtype == torch.float64 and mat64.dtype == torch.float64
    assert torch.allclose(mat64, mat32, atol=1e-6), (mat64 - mat32).abs().max().item()
    assert keep32.tolist() == keep64.tolist() and mask32.tolist() == mask64.tolist()


def check_errors(solve):
    good = _boxes([[0, 0, 2, 2, 0]])
    good_s = _scores([0.9])

    def bad(fn, desc):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError(f"应抛 ValueError: {desc}")

    bad(lambda: solve([[0, 0, 2, 2, 0]], good_s), "boxes 非张量")
    bad(lambda: solve(good, [0.9]), "scores 非张量")
    bad(lambda: solve(torch.zeros(1, 4), good_s), "boxes 列数不为 5")
    bad(lambda: solve(torch.zeros(5), good_s), "boxes 非二维")
    bad(lambda: solve(good, torch.zeros(1, 1)), "scores 非一维")
    bad(lambda: solve(good, torch.zeros(2)), "scores 长度不符")
    bad(lambda: solve(good.to(torch.int64), good_s.to(torch.int64)), "dtype 非浮点")
    bad(lambda: solve(good, good_s.to(torch.float32)), "boxes/scores dtype 不一致")
    bad(lambda: solve(_boxes([[0, 0, 0, 2, 0]]), good_s), "w 为 0")
    bad(lambda: solve(_boxes([[0, 0, 2, 0, 0]]), good_s), "h 为 0")
    bad(lambda: solve(_boxes([[0, 0, -1, 2, 0]]), good_s), "w 为负")
    bad(lambda: solve(_boxes([[float("nan"), 0, 2, 2, 0]]), good_s), "boxes 含 NaN")
    bad(lambda: solve(good, _scores([float("inf")])), "scores 含 Inf")
    bad(lambda: solve(good, good_s, iou_thr=-0.1), "iou_thr 为负")
    bad(lambda: solve(good, good_s, iou_thr=1.5), "iou_thr 超 1")
