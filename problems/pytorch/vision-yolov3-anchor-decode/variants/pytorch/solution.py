"""YOLOv3 anchor 系 box 编解码账（solution）。

转写 pjreddie/darknet f86901f6 src/yolo_layer.c：
get_yolo_box / delta_yolo_box / forward 中的 best-anchor 匹配。
"""

import math


def _sigmoid(v):
    if v >= 0:
        return 1.0 / (1.0 + math.exp(-v))
    e = math.exp(v)
    return e / (1.0 + e)


def _iou_wh(w1, h1, w2, h2):
    # 中心对齐（darknet box.c：两框中心重合时 inter=min·min）
    if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
        return 0.0
    inter = min(w1, w2) * min(h1, h2)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union


def solve(lw, lh, img_w, img_h, anchors, truths, preds):
    if not isinstance(lw, int) or not isinstance(lh, int) or lw < 1 or lh < 1:
        raise ValueError("lw, lh must be int >= 1")
    for v in (img_w, img_h):
        if not isinstance(v, (int, float)) or v <= 0:
            raise ValueError("img_w, img_h must be positive")
    if not anchors or any(
        not isinstance(a, list) or len(a) != 2 or a[0] <= 0 or a[1] <= 0
        for a in anchors
    ):
        raise ValueError("anchors must be [[pw, ph], ...] positive")
    if len(truths) != len(preds):
        raise ValueError("truths and preds must be aligned")
    for t in truths:
        if not isinstance(t, list) or len(t) != 4 or not all(
            isinstance(v, (int, float)) and math.isfinite(v) for v in t
        ):
            raise ValueError("each truth must be [x, y, w, h] finite")
        if t[2] <= 0 or t[3] <= 0:
            raise ValueError("truth w,h must be positive")
    for p in preds:
        if not isinstance(p, list) or len(p) != 4 or not all(
            isinstance(v, (int, float)) and math.isfinite(v) for v in p
        ):
            raise ValueError("each pred must be [tx, ty, tw, th] finite raw logits")

    matches = []
    for t_idx, (truth, raw) in enumerate(zip(truths, preds)):
        tx_raw, ty_raw, tw_raw, th_raw = raw
        # forward 中的 best-anchor 匹配：truth 中心移原点，wh-only IoU
        best_n = 0
        best_iou = 0.0
        for n, (pw, ph) in enumerate(anchors):
            iou = _iou_wh(pw / img_w, ph / img_h, truth[2], truth[3])
            if iou > best_iou:          # 严格 >：并列取小下标
                best_iou = iou
                best_n = n
        pw_px, ph_px = anchors[best_n]

        # 落格：int 截断，官方无 clamp
        i = int(truth[0] * lw)
        j = int(truth[1] * lh)

        # encode target（delta_yolo_box）：xy 是线性偏移（非 logit 逆），
        # wh 是 log 比（与 decode 的 exp 互逆）
        tgt_x = truth[0] * lw - i
        tgt_y = truth[1] * lh - j
        tgt_w = math.log(truth[2] * img_w / pw_px)
        tgt_h = math.log(truth[3] * img_h / ph_px)

        # decode（get_yolo_box）：σ(tx)+i 归一化、exp(tw)·anchor/图尺寸
        pred_x = (i + _sigmoid(tx_raw)) / lw
        pred_y = (j + _sigmoid(ty_raw)) / lh
        pred_w = math.exp(tw_raw) * pw_px / img_w
        pred_h = math.exp(th_raw) * ph_px / img_h

        # delta：scale = 2 − truth.w·truth.h（truth 归一化值）
        scale = 2.0 - truth[2] * truth[3]
        delta = [
            scale * (tgt_x - _sigmoid(tx_raw)),
            scale * (tgt_y - _sigmoid(ty_raw)),
            scale * (tgt_w - tw_raw),
            scale * (tgt_h - th_raw),
        ]

        matches.append({
            "truth_index": t_idx,
            "anchor": best_n,
            "cell": [i, j],
            "target": [tgt_x, tgt_y, tgt_w, tgt_h],
            "pred_box": [pred_x, pred_y, pred_w, pred_h],
            "delta": delta,
            "anchor_iou": best_iou,
        })

    return {"matches": matches}
