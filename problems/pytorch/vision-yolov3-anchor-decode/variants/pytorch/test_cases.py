"""YOLOv3 anchor 系 box 编解码账（test cases）。

独立 oracle：函数式直算（与 solution 的循环重放不同组织）。
"""

import math
import random


def _sig(v):
    return 1.0 / (1.0 + math.exp(-v)) if v >= 0 else math.exp(v) / (1.0 + math.exp(v))


def _py_solve(lw, lh, img_w, img_h, anchors, truths, preds):
    out = []
    for t_idx, (truth, raw) in enumerate(zip(truths, preds)):
        # best anchor：wh-only 中心对齐 IoU，严格 > 取小下标
        ious = []
        for pw, ph in anchors:
            nw, nh = pw / img_w, ph / img_h
            inter = min(nw, truth[2]) * min(nh, truth[3])
            union = nw * nh + truth[2] * truth[3] - inter
            ious.append(inter / union if union > 0 else 0.0)
        best_n = max(range(len(anchors)), key=lambda n: (ious[n], -n))
        pw_px, ph_px = anchors[best_n]
        i, j = int(truth[0] * lw), int(truth[1] * lh)
        tgt = [truth[0] * lw - i, truth[1] * lh - j,
               math.log(truth[2] * img_w / pw_px), math.log(truth[3] * img_h / ph_px)]
        pred = [(i + _sig(raw[0])) / lw, (j + _sig(raw[1])) / lh,
                math.exp(raw[2]) * pw_px / img_w, math.exp(raw[3]) * ph_px / img_h]
        scale = 2.0 - truth[2] * truth[3]
        outp = [_sig(raw[0]), _sig(raw[1]), raw[2], raw[3]]
        delta = [scale * (t - o) for t, o in zip(tgt, outp)]
        out.append({"truth_index": t_idx, "anchor": best_n, "cell": [i, j],
                    "target": tgt, "pred_box": pred, "delta": delta,
                    "anchor_iou": ious[best_n]})
    return {"matches": out}


def _close(actual, expect, tol=1e-9):
    if isinstance(expect, dict):
        assert isinstance(actual, dict)
        assert len(actual) == len(expect)
        return all(_close(actual[k], expect[k], tol) for k in expect)
    if isinstance(expect, list):
        assert isinstance(actual, list) and len(actual) == len(expect), \
            f"{actual!r} vs {expect!r}"
        return all(_close(a, e, tol) for a, e in zip(actual, expect))
    if isinstance(expect, float):
        assert abs(actual - expect) < tol, f"{actual} != {expect}"
        return True
    assert actual == expect, f"{actual!r} != {expect!r}"
    return True


def _compare(solve, lw, lh, img_w, img_h, anchors, truths, preds):
    led = solve(lw, lh, img_w, img_h, anchors, truths, preds)
    _close(led, _py_solve(lw, lh, img_w, img_h, anchors, truths, preds))
    return led


def check_basic_single(solve):
    # 单 anchor：σ(0)=0.5 → bx=(3+0.5)/7；wh：e^0·pw/W
    led = _compare(solve, 7, 7, 416, 416, [[32.0, 64.0]],
                   [[0.55, 0.6, 0.2, 0.3]], [[0.0, 0.0, 0.0, 0.0]])
    m = led["matches"][0]
    assert m["anchor"] == 0
    assert m["cell"] == [3, 4]
    _close(m["pred_box"][0], 3.5 / 7)
    _close(m["pred_box"][2], 32.0 / 416)
    # encode target：xy 线性偏移、wh log 比
    _close(m["target"][0], 0.85)
    _close(m["target"][2], math.log(0.2 * 416 / 32.0))
    # wh 往返互逆：e^tgt·pw/W == truth.w
    _close(math.exp(m["target"][2]) * 32.0 / 416, 0.2)
    # scale = 2 − w·h；xy delta 含 (tgt − σ(0)) = tgt − 0.5
    _close(m["delta"][0], (2 - 0.2 * 0.3) * (0.85 - 0.5))
    _close(m["delta"][2], (2 - 0.2 * 0.3) * m["target"][2])
    # anchor_iou 的精确值由 basic_anchor_iou_value 覆盖


def check_basic_anchor_iou_value(solve):
    # anchor (32,64)@416 → (0.0769, 0.1538)；truth (0.2,0.3)
    led = _compare(solve, 7, 7, 416, 416, [[32.0, 64.0]],
                   [[0.55, 0.6, 0.2, 0.3]], [[0.0, 0.0, 0.0, 0.0]])
    m = led["matches"][0]
    nw, nh = 32 / 416, 64 / 416
    inter = min(nw, 0.2) * min(nh, 0.3)
    union = nw * nh + 0.2 * 0.3 - inter
    _close(m["anchor_iou"], inter / union)


def check_anchor_tie_lower_index(solve):
    # 两个相同 anchor：并列严格 > → 取小下标
    led = _compare(solve, 7, 7, 416, 416,
                   [[32.0, 32.0], [32.0, 32.0]],
                   [[0.5, 0.5, 0.1, 0.1]], [[0.0] * 4])
    assert led["matches"][0]["anchor"] == 0


def check_anchor_best_wins(solve):
    # 三 anchor：wh 最接近的赢（严格 >，首个最大者）
    led = _compare(solve, 7, 7, 416, 416,
                   [[400.0, 400.0], [80.0, 80.0], [80.0, 40.0]],
                   [[0.5, 0.5, 0.1, 0.2]], [[0.0] * 4])
    assert led["matches"][0]["anchor"] == 1    # 80/416=0.192 vs truth 0.1,0.2
    # 精确相等时后一个不大于是 → 小下标
    led = _compare(solve, 7, 7, 416, 416,
                   [[83.2, 83.2], [83.2, 83.2]],
                   [[0.5, 0.5, 0.2, 0.2]], [[0.0] * 4])
    assert led["matches"][0]["anchor"] == 0


def check_xy_not_invertible(solve):
    # xy encode target 是线性偏移；与 decode σ 不可逆：
    # raw 使 σ(tx)=tgt_x → delta xy = 0，但 raw ≠ logit(逆) 官方不提供逆
    lw = 4
    truth = [0.375, 0.625, 0.2, 0.2]           # i=1, j=2; tgt_x=0.5
    led = _compare(solve, lw, lw, 416, 416, [[64.0, 64.0]],
                   [truth], [[0.0, 0.0, 0.0, 0.0]])
    m = led["matches"][0]
    assert m["cell"] == [1, 2]
    _close(m["target"][0], 0.5)
    _close(m["delta"][0], 0.0)                 # σ(0)=0.5 == tgt_x
    # wh delta ≠ 0（raw 0 vs log 比：log(0.2·416/64)=log 1.3）
    assert abs(m["delta"][2]) > 0.1


def check_decode_shapes(solve):
    # σ(logit=0.3544...)≈0.5877；exp(0.6931)=2 → bw=2·pw/W
    raw = [0.35439384978317273, -0.8472978603872037, math.log(2.0), math.log(0.5)]
    led = _compare(solve, 7, 7, 416, 416, [[32.0, 32.0]],
                   [[0.55, 0.6, 0.2, 0.2]], [raw])
    m = led["matches"][0]
    _close(m["pred_box"][0], (3 + 0.58775) / 7, tol=1e-4)
    _close(m["pred_box"][2], 2 * 32.0 / 416)
    _close(m["pred_box"][3], 0.5 * 32.0 / 416)


def check_oracle_random(solve):
    rng = random.Random(702)
    for _ in range(250):
        lw = rng.choice([4, 7, 13])
        lh = rng.choice([4, 7, 13])
        img_w = rng.choice([320, 416, 608])
        img_h = img_w
        na = rng.randint(1, 5)
        anchors = [[round(rng.uniform(16, 320), 1), round(rng.uniform(16, 320), 1)]
                   for _ in range(na)]
        nt = rng.randint(1, 4)
        truths = [[round(rng.uniform(0.05, 0.95), 3), round(rng.uniform(0.05, 0.95), 3),
                   round(rng.uniform(0.02, 0.6), 3), round(rng.uniform(0.02, 0.6), 3)]
                  for _ in range(nt)]
        preds = [[round(rng.uniform(-2, 2), 3) for _ in range(4)] for _ in range(nt)]
        _compare(solve, lw, lh, img_w, img_h, anchors, truths, preds)


def check_invalid_inputs(solve):
    base = dict(lw=7, lh=7, img_w=416, img_h=416,
                anchors=[[32.0, 32.0]], truths=[[0.5, 0.5, 0.2, 0.2]],
                preds=[[0.0] * 4])
    def call(**kw):
        args = {**base, **kw}
        return solve(args["lw"], args["lh"], args["img_w"], args["img_h"],
                     args["anchors"], args["truths"], args["preds"])
    for kw in [
        dict(lw=0), dict(lh=-1), dict(lw=1.5),
        dict(img_w=0), dict(img_h=-1),
        dict(anchors=[]), dict(anchors=[[0.0, 32.0]]), dict(anchors=[[32.0]]),
        dict(truths=[[0.5, 0.5, 0.2]]),
        dict(truths=[[0.5, 0.5, 0.0, 0.2]]),
        dict(truths=[[float("nan"), 0.5, 0.2, 0.2]]),
        dict(preds=[[0.0] * 3]),
        dict(truths=[], preds=[[0.0] * 4]),     # 长度不齐
    ]:
        try:
            call(**kw)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError(f"expected error: {kw}")
