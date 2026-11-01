"""YOLOv4 Mosaic 增强 truth 侧账（solution）。

转写 AlexeyAB/darknet src/data.c blend_truth_mosaic + load_data_detection
的 mosaic 图像拷贝分支（shifts/cut 显式入参，不考随机）。
"""

import math


def solve(w, h, cut_x, cut_y, i_mixup, shifts, old_truths,
          mosaic_bound=False):
    for name, v in (("w", w), ("h", h)):
        if not isinstance(v, int) or v < 1:
            raise ValueError(f"{name} must be int >= 1")
    for name, v, lim in (("cut_x", cut_x, w), ("cut_y", cut_y, h)):
        if not isinstance(v, int) or not 0 <= v <= lim:
            raise ValueError(f"{name} must be int in [0, {lim}]")
    if i_mixup not in (0, 1, 2, 3):
        raise ValueError("i_mixup must be 0..3")
    if (not isinstance(shifts, list) or len(shifts) != 4 or not all(
        isinstance(s, int) and s >= 0 for s in shifts
    )):
        raise ValueError("shifts must be 4 non-negative ints")
    left_shift, right_shift, top_shift, bot_shift = shifts
    for t in old_truths:
        if not isinstance(t, list) or len(t) != 5 or not all(
            isinstance(v, (int, float)) and math.isfinite(v) for v in t
        ):
            raise ValueError("each truth must be [x, y, w, h, id] finite")
        if t[2] <= 0 or t[3] <= 0:
            raise ValueError("truth w,h must be positive")
    if not isinstance(mosaic_bound, bool):
        raise ValueError("mosaic_bound must be bool")

    lowest_w = 1.0 / w
    lowest_h = 1.0 / h
    boxes = []
    dropped = []

    for idx, t in enumerate(old_truths):
        xb, yb, wb, hb = (float(v) for v in t[:4])
        tid = t[4]

        # 象限平移（四分支方向不对称）
        if i_mixup == 0:
            xb -= (w - cut_x - right_shift) / w
            yb -= (h - cut_y - bot_shift) / h
        elif i_mixup == 1:
            xb += (cut_x - left_shift) / w
            yb -= (h - cut_y - bot_shift) / h
        elif i_mixup == 2:
            xb -= (w - cut_x - right_shift) / w
            yb += (cut_y - top_shift) / h
        else:
            xb += (cut_x - left_shift) / w
            yb += (cut_y - top_shift) / h

        # 像素角（C int 截断）
        left = int((xb - wb / 2) * w)
        right = int((xb + wb / 2) * w)
        top = int((yb - hb / 2) * h)
        bot = int((yb + hb / 2) * h)

        if mosaic_bound:
            # 按象限界 clamp（注意 left<bound 才收、right>bound 才收）
            if i_mixup == 0:
                lb, rb, tb, bb = 0, cut_x, 0, cut_y
            elif i_mixup == 1:
                lb, rb, tb, bb = cut_x, w, 0, cut_y
            elif i_mixup == 2:
                lb, rb, tb, bb = 0, cut_x, cut_y, h
            else:
                lb, rb, tb, bb = cut_x, w, cut_y, h
            if left < lb:
                left = lb
            if right > rb:
                right = rb
            if top < tb:
                top = tb
            if bot > bb:
                bot = bb
            # clamp 后由角重算
            xb = (right + left) / 2 / w
            wb = (right - left) / w
            yb = (bot + top) / 2 / h
            hb = (bot - top) / h
        else:
            # 越界侧收缩修正（先于终门，可把越界框救回）：
            # left<0 → diff=left/w<0, xb−=diff/2（右移）, wb+=diff（收缩）
            if left < 0:
                diff = left / w
                xb -= diff / 2
                wb += diff
            if right > w:
                diff = (right - w) / w
                xb -= diff / 2
                wb -= diff
            if top < 0:
                diff = top / h
                yb -= diff / 2
                hb += diff
            if bot > h:
                diff = (bot - h) / h
                yb -= diff / 2
                hb -= diff
            left = int((xb - wb / 2) * w)
            right = int((xb + wb / 2) * w)
            top = int((yb - hb / 2) * h)
            bot = int((yb + hb / 2) * h)

        # 终门（单 AND）：像素界 + 归一化半开 + 0<w<1 + lowest
        ok = (left >= 0 and right <= w and top >= 0 and bot <= h
              and wb > 0 and wb < 1 and hb > 0 and hb < 1
              and xb > 0 and xb < 1 and yb > 0 and yb < 1
              and wb > lowest_w and hb > lowest_h)
        if ok:
            boxes.append([xb, yb, wb, hb, tid])
        else:
            dropped.append(idx)

    # 当前象限的图像拷贝账（memcpy 四分支）
    if i_mixup == 0:
        plan = {"dst": [0, 0], "src": [w - cut_x - right_shift,
                                       h - cut_y - bot_shift],
                "size": [cut_x, cut_y]}
    elif i_mixup == 1:
        plan = {"dst": [cut_x, 0], "src": [left_shift, h - cut_y - bot_shift],
                "size": [w - cut_x, cut_y]}
    elif i_mixup == 2:
        plan = {"dst": [0, cut_y], "src": [w - cut_x - right_shift, top_shift],
                "size": [cut_x, h - cut_y]}
    else:
        plan = {"dst": [cut_x, cut_y], "src": [left_shift, top_shift],
                "size": [w - cut_x, h - cut_y]}

    return {"boxes": boxes, "dropped_indices": dropped, "copy_plan": plan}
