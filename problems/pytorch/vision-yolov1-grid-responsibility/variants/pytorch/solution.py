"""YOLOv1 网格责任编码账（solution）。

转写 pjreddie/darknet commit 9db61832（2015-07，YOLOv1 CVPR 期）：
src/data.c fill_truth_detection + src/detection.c convert_detections。
"""

import math


def _constrain01(v):
    return min(1.0, max(0.0, v))


def solve(side, classes, labels, jitter=None, flip=False):
    if not isinstance(side, int) or side < 1:
        raise ValueError("side must be int >= 1")
    if not isinstance(classes, int) or classes < 1:
        raise ValueError("classes must be int >= 1")
    if jitter is None:
        jitter = [0.0, 0.0, 1.0, 1.0]
    if not isinstance(jitter, list) or len(jitter) != 4 or not all(
        isinstance(v, (int, float)) and math.isfinite(v) for v in jitter
    ):
        raise ValueError("jitter must be [dx, dy, sx, sy] finite floats")
    if not isinstance(flip, bool):
        raise ValueError("flip must be bool")
    for lab in labels:
        if not isinstance(lab, list) or len(lab) != 5:
            raise ValueError("each label must be [id, x, y, w, h]")
        lid, x, y, w, h = lab
        if not isinstance(lid, int) or not 0 <= lid < classes:
            raise ValueError("label id out of range")
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (x, y, w, h)):
            raise ValueError("label coords must be finite floats")
        if w <= 0 or h <= 0:
            raise ValueError("label w,h must be positive")

    dx, dy, sx, sy = [float(v) for v in jitter]
    grid = {}
    decoded = {}
    dropped = []

    for idx, lab in enumerate(labels):
        lid, x0, y0, w0, h0 = lab
        # read_boxes 角展开 → jitter 变换 → flip → constrain → 重新求中心/宽高
        left = (x0 - w0 / 2.0) * sx - dx
        right = (x0 + w0 / 2.0) * sx - dx
        top = (y0 - h0 / 2.0) * sy - dy
        bot = (y0 + h0 / 2.0) * sy - dy
        if flip:
            left, right = 1.0 - right, 1.0 - left
        left = _constrain01(left)
        right = _constrain01(right)
        top = _constrain01(top)
        bot = _constrain01(bot)
        x = (left + right) / 2.0
        y = (top + bot) / 2.0
        w = right - left          # 截断后宽高，不是原值 clamp
        h = bot - top

        # 中心半开门：恰在 0 或 1 都丢弃
        if x <= 0 or x >= 1 or y <= 0 or y >= 1:
            dropped.append({"index": idx, "reason": "center_out"})
            continue

        # 网格分配：int 截断（中心门保证 col,row ∈ [0, side-1]，无 clamp）
        col = int(x * side)
        row = int(y * side)
        tx = x * side - col
        ty = y * side - row

        # 宽高门（截断后）：严格 <
        if w < 0.01 or h < 0.01:
            dropped.append({"index": idx, "reason": "too_small"})
            continue

        # sqrt 编码（官方 pow(w, 1./2.)）
        tw = w ** 0.5
        th = h ** 0.5

        key = f"{row},{col}"
        if key in grid:
            # 同格先到先得：truth 槽已占用则丢弃后来 GT
            dropped.append({"index": idx, "reason": "occupied"})
            continue
        grid[key] = {"id": lid, "x": tx, "y": ty, "w": tw, "h": th, "conf": 1.0}
        # convert_detections 互逆 decode：平移 + 平方
        decoded[key] = [
            (col + tx) / side,
            (row + ty) / side,
            tw * tw,
            th * th,
        ]

    return {"grid": grid, "decoded": decoded, "dropped": dropped}
