"""order 718: PaddleOCR DB 后处理单框账（box_score_fast + unclip + 门序）。

oracle 用 numpy 向量化独立实现同一官方语义（与 solve 的标量循环不同
写法）；锚点全部官方库实测钉死（shapely 2.2 / pyclipper 1.4 / cv2 5.0，
见 notes）。
"""
import math
import random

import numpy as np


# ---------------- oracle（numpy 独立实现） ----------------

def _oracle(pred, box, thresh=0.3, box_thresh=0.7, unclip_ratio=2.0,
            min_size=3, dest_width=None, dest_height=None):
    P = np.asarray(pred, dtype=np.float64)
    H, W = P.shape
    B = np.asarray(box, dtype=np.float64)
    if B.shape != (4, 2):
        raise ValueError("box must be 4 points of [x, y]")
    assert (P < 0).sum() == 0 and (P > 1).sum() == 0

    bitmap = (P > thresh).tolist()
    pos_count = int((P > thresh).sum())

    x_lo, x_hi = float(B[:, 0].min()), float(B[:, 0].max())
    y_lo, y_hi = float(B[:, 1].min()), float(B[:, 1].max())
    sside_in = min(x_hi - x_lo, y_hi - y_lo)
    out = {"bitmap": bitmap, "pos_count": pos_count,
           "sside_in": sside_in, "score": None, "bbox": None,
           "distance": None, "expanded": None, "sside_out": None,
           "gate": None, "passed": False, "scaled": None}
    if sside_in < min_size:
        out["gate"] = "min_size_in"
        return out

    xmin = int(np.clip(np.floor(x_lo), 0, W - 1))
    xmax = int(np.clip(np.ceil(x_hi), 0, W - 1))
    ymin = int(np.clip(np.floor(y_lo), 0, H - 1))
    ymax = int(np.clip(np.ceil(y_hi), 0, H - 1))
    out["bbox"] = [xmin, xmax, ymin, ymax]
    # fillPoly: astype(int32) 截断后闭矩形（含右/下边界），与 bbox 求交
    tx_lo, tx_hi = int(B[:, 0].min()), int(B[:, 0].max())   # astype 截断
    ty_lo, ty_hi = int(B[:, 1].min()), int(B[:, 1].max())
    xx, yy = np.meshgrid(np.arange(xmin, xmax + 1),
                         np.arange(ymin, ymax + 1))
    mask = (xx >= tx_lo) & (xx <= tx_hi) & (yy >= ty_lo) & (yy <= ty_hi)
    score = float(P[ymin:ymax + 1, xmin:xmax + 1][mask].mean()) \
        if mask.any() else 0.0
    out["score"] = score
    if box_thresh > score:
        out["gate"] = "box_thresh"
        return out

    d = B - np.roll(B, -1, axis=0)          # 闭合边向量（含末->首）
    perim = float(np.hypot(d[:, 0], d[:, 1]).sum())
    if perim == 0:
        raise ValueError("degenerate box (zero perimeter)")
    # 独立写法：sum(x_i*y_{i+1} - x_{i+1}*y_i) / 2
    bx1 = B[:, 0]; by1 = B[:, 1]
    bx2 = np.roll(B[:, 0], -1); by2 = np.roll(B[:, 1], -1)
    area = float(np.sum(bx1 * by2 - bx2 * by1)) / 2.0
    distance = abs(area) * unclip_ratio / perim

    def rnd(v):                              # pyclipper: round half away
        return math.floor(v + 0.5)

    bx_lo = rnd(int(x_lo) - distance)
    bx_hi = rnd(int(x_hi) + distance)
    by_lo = rnd(int(y_lo) - distance)
    by_hi = rnd(int(y_hi) + distance)
    expanded = [[bx_lo, by_lo], [bx_hi, by_lo],
                [bx_hi, by_hi], [bx_lo, by_hi]]
    sside_out = min(bx_hi - bx_lo, by_hi - by_lo)
    out["distance"] = distance
    out["expanded"] = expanded
    out["sside_out"] = sside_out
    if sside_out < min_size + 2:
        out["gate"] = "min_size_out"
        return out
    out["passed"] = True
    if dest_width is not None and dest_height is not None:
        E = np.asarray(expanded, dtype=np.float64)
        sx = np.clip(np.rint(E[:, 0] / W * dest_width), 0, dest_width)
        sy = np.clip(np.rint(E[:, 1] / H * dest_height), 0, dest_height)
        out["scaled"] = [[int(a), int(b)] for a, b in zip(sx, sy)]
    return out


def _compare(solve, pred, box, **kw):
    led = solve([list(r) for r in pred], [list(p) for p in box], **kw)
    exp = _oracle([list(r) for r in pred], [list(p) for p in box], **kw)
    assert set(led) == set(exp)
    for k in exp:
        lv, ev = led[k], exp[k]
        if k in ("score", "distance") and lv is not None and ev is not None:
            assert math.isclose(lv, ev, rel_tol=0, abs_tol=1e-9), \
                (k, lv, ev)
        else:
            assert lv == ev, (k, lv, ev)
    return led


def _const_pred(h, w, region_val, region, base=0.0):
    pred = [[base] * w for _ in range(h)]
    for y in range(region[2], region[3] + 1):
        for x in range(region[0], region[1] + 1):
            pred[y][x] = region_val
    return pred


def check_basic(solve):
    """手算锚点：box [1,1]-[4,4]、ratio 2 -> dist 1.5、expanded [0,6]^2。"""
    pred = _const_pred(5, 6, 0.75, (1, 4, 1, 4))
    led = _compare(solve, pred, [[1, 1], [4, 1], [4, 4], [1, 4]])
    assert led["pos_count"] == 16
    assert led["sside_in"] == 3.0
    assert led["bbox"] == [1, 4, 1, 4]
    assert led["score"] == 0.75
    assert math.isclose(led["distance"], 1.5)
    assert led["expanded"] == [[0, 0], [6, 0], [6, 6], [0, 6]]
    assert led["sside_out"] == 6
    assert led["gate"] is None and led["passed"]


def check_cw_order(solve):
    """顺时针顶点序：|shoelace| 保证 distance 不变。"""
    pred = _const_pred(5, 6, 0.75, (1, 4, 1, 4))
    a = _compare(solve, pred, [[1, 1], [4, 1], [4, 4], [1, 4]])
    b = _compare(solve, pred, [[1, 4], [4, 4], [4, 1], [1, 1]])   # cw
    assert math.isclose(a["distance"], b["distance"]) and a["passed"]
    assert a["expanded"] == b["expanded"]


def check_thresh_strict(solve):
    """pred 恰等于 thresh 不算正点（严格 >）。"""
    pred = _const_pred(5, 6, 0.75, (1, 4, 1, 4), base=0.3)
    led = _compare(solve, pred, [[1, 1], [4, 1], [4, 4], [1, 4]])
    assert led["pos_count"] == 16          # base 0.3 == thresh 不计


def check_score_bbox_vs_trunc(solve):
    """浮点矩形：bbox floor/ceil 与 mask 截断矩形两套坐标共存。"""
    pred = _const_pred(8, 12, 0.75, (2, 6, 1, 5))
    led = _compare(solve, pred, [[2.3, 1.7], [6.4, 1.7], [6.4, 5.2],
                                 [2.3, 5.2]])
    assert led["bbox"] == [2, 7, 1, 6]        # ceil(6.4)=7, ceil(5.2)=6
    assert led["score"] == 0.75                # trunc mask [2,6]x[1,5] 全含
    assert math.isclose(led["distance"], 14.35 * 2.0 / 15.2)


def check_negative_coords(solve):
    """负坐标：bbox clip 到 0、fillPoly 截断向零（-3.7 -> -3 非 floor）。"""
    pred = _const_pred(6, 6, 0.75, (0, 1, 0, 2))
    led = _compare(solve, pred, [[-3.7, -2.2], [1.4, -2.2], [1.4, 2.8],
                                 [-3.7, 2.8]])
    assert led["bbox"] == [0, 2, 0, 3]
    assert led["score"] == 0.75                # mask 与 bbox 交 2x3 全在区内
    assert math.isclose(led["distance"], 25.5 * 2.0 / 20.2)


def check_clip_edge(solve):
    """贴右边界：xmax=13 被 clip 到 W-1=11。"""
    pred = _const_pred(8, 14, 0.75, (10, 13, 0, 7))
    led = _compare(solve, pred, [[10, 0], [13, 0], [13, 7], [10, 7]])
    assert led["bbox"] == [10, 13, 0, 7]
    assert led["score"] == 0.75


def check_outside_zero_mask(solve):
    """框完全出图左侧：mask 交空 -> cv2.mean 语义 score=0.0 -> box 门。"""
    pred = _const_pred(8, 12, 0.75, (0, 1, 0, 4))
    led = _compare(solve, pred, [[-8.5, -0.5], [-4.2, -0.5],
                                 [-4.2, 3.5], [-8.5, 3.5]])
    assert led["bbox"] == [0, 0, 0, 4]
    assert led["score"] == 0.0
    assert led["gate"] == "box_thresh" and not led["passed"]
    assert led["expanded"] is None            # 官方顺序：score 门后不算


def check_min_size_in(solve):
    """单点退化：sside_in=0 < 3 -> score/distance 不算。"""
    pred = _const_pred(8, 12, 0.9, (3, 3, 3, 3))
    led = _compare(solve, pred, [[3.0, 3.0]] * 4)
    assert led["gate"] == "min_size_in" and led["score"] is None
    assert not led["passed"]


def check_sside_out_gate(solve):
    """unclip 后仍小于 min_size+2（细长框 + 小 ratio）。"""
    pred = _const_pred(8, 20, 0.75, (0, 12, 0, 3))
    led = _compare(solve, pred, [[0, 0], [12, 0], [12, 3], [0, 3]],
                   unclip_ratio=0.05)
    assert led["gate"] == "min_size_out"
    assert led["sside_out"] == 3              # y span: round(-0.06)..round(3.06)
    assert not led["passed"]


def check_sside_out_boundary(solve):
    """sside_out 恰为 4：+1 变异（span 误用像素数）跨界用例。"""
    pred = _const_pred(8, 20, 0.75, (0, 8, 0, 4))
    led = _compare(solve, pred, [[0, 0], [8, 0], [8, 4], [0, 4]],
                   unclip_ratio=0.225)
    assert led["gate"] == "min_size_out"
    assert led["sside_out"] == 4


def check_box_thresh_tie(solve):
    """score 恰等于 box_thresh：官方严格 > 过门。"""
    pred = _const_pred(5, 6, 0.5, (1, 4, 1, 4))
    led = _compare(solve, pred, [[1, 1], [4, 1], [4, 4], [1, 4]],
                   box_thresh=0.5)
    assert led["gate"] is None and led["passed"]


def check_dest_scaling(solve):
    """dest 缩放：银行家舍入（6.5->6、2.5->2）+ 双界 clip。"""
    pred = _const_pred(8, 4, 0.75, (0, 3, 0, 3))
    led = _compare(solve, pred, [[0, 0], [3, 0], [3, 3], [0, 3]],
                   dest_width=10, dest_height=10)
    assert led["expanded"] == [[-1, -1], [5, -1], [5, 5], [-1, 5]]
    assert led["scaled"] == [[0, 0], [10, 0], [10, 6], [0, 6]]


def check_dest_bankers(solve):
    """银行家舍入锚点：x/W*dw = 1/6*3 = 0.5 -> 0（half-even），
    half-up 变异会给 1。"""
    pred = _const_pred(6, 6, 0.75, (0, 0, 0, 0))
    led = _compare(solve, pred, [[-1, -1], [0, -1], [0, 0], [-1, 0]],
                   unclip_ratio=4.0, min_size=0,
                   dest_width=3, dest_height=3)
    # expanded: bx_lo=round(-1-1)=-2, bx_hi=round(0+1)=1 (同 y)
    assert led["expanded"] == [[-2, -2], [1, -2], [1, 1], [-2, 1]]
    # sx: -2/6*3=-1 -> clip 0; 1/6*3=0.5 -> banker 0
    assert led["scaled"] == [[0, 0], [0, 0], [0, 0], [0, 0]]


def check_invalid_inputs(solve):
    pred = _const_pred(5, 6, 0.75, (1, 4, 1, 4))
    bad = [
        lambda: solve(pred, [[1, 1], [4, 1], [4, 4]]),           # 3 点
        lambda: solve(pred, [[1, 1, 0], [4, 1], [4, 4], [1, 4]]),  # 维度
        lambda: solve(pred, [[3.0, 3.0]] * 4, min_size=0),       # perim 0
    ]
    for fn in bad:
        try:
            fn()
        except (ValueError, ZeroDivisionError):
            pass
        else:
            raise AssertionError("expected error")


def check_randomized(solve):
    rng = random.Random(718)
    for _ in range(200):
        h = rng.randint(4, 12)
        w = rng.randint(4, 16)
        pred = [[round(rng.random(), 3) for _ in range(w)]
                for _ in range(h)]
        half = rng.random() < 0.3
        x0 = rng.randint(-3, w - 2)
        x1 = x0 + rng.randint(1, 6)
        y0 = rng.randint(-3, h - 2)
        y1 = y0 + rng.randint(1, 5)
        if half:
            x0 += 0.5; x1 += 0.5; y0 += 0.5; y1 += 0.5
        box = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        kw = dict(thresh=round(rng.uniform(0.1, 0.5), 2),
                  box_thresh=round(rng.uniform(0.3, 0.9), 2),
                  unclip_ratio=round(rng.uniform(0.3, 3.0), 2),
                  min_size=rng.randint(0, 4))
        if rng.random() < 0.5:
            kw["dest_width"] = rng.randint(2, 30)
            kw["dest_height"] = rng.randint(2, 30)
        # 半整数坐标会让 pyclipper 输入截断语义含糊（notes），
        # oracle 与 solve 同语义但对拍仍须一致——半整数时跳过 perim 0
        try:
            _compare(solve, pred, box, **kw)
        except (ValueError, ZeroDivisionError):
            pass


def check_no_forbidden_calls(solve):
    """红线：不得 import cv2 / shapely / pyclipper（AST 节点级）。"""
    import ast
    import os
    import sys
    mod = sys.modules.get(__name__)
    if mod is None or not getattr(mod, "__file__", None):
        return
    path = os.path.join(os.path.dirname(mod.__file__), "solution.py")
    tree = ast.parse(open(path).read())
    banned = {"cv2", "shapely", "pyclipper"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in banned:
                    raise AssertionError(f"banned import: {a.name}")
        if isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in banned:
                raise AssertionError(f"banned import: {node.module}")
