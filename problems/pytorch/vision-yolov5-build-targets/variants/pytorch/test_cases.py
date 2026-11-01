"""YOLOv5 build_targets 分配账（test cases）。

独立 oracle：按 (anchor, dir) 组合直接枚举（无先复制后过滤结构）
+ 200 组随机对拍。
"""

import math
import random


def _oracle(targets, anchors, shapes, anchor_t):
    nl = len(anchors)
    na = len(anchors[0])
    g = 0.5
    offs = [(0.0, 0.0), (g, 0.0), (0.0, g), (-g, 0.0), (0.0, -g)]
    layers = []
    for i in range(nl):
        W, H = float(shapes[i][1]), float(shapes[i][0])
        nt = len(targets)
        keep = []
        for a in range(na):
            for ti in range(nt):
                t = targets[ti]
                ok = True
                for dim in range(2):
                    gw = t[3] * W if dim == 0 else t[4] * H
                    aw = anchors[i][a][dim]
                    r = gw / aw
                    if max(r, 1 / r) >= anchor_t:
                        ok = False
                        break
                keep.append(ok)
        rows = []
        for a in range(na):
            for ti in range(nt):
                if not keep[a * nt + ti]:
                    continue
                t = targets[ti]
                gx, gy = t[1] * W, t[2] * H
                gwh = (t[3] * W, t[4] * H)
                gxi, gyi = W - gx, H - gy
                dirs = [True,
                        (gx % 1 < g) and (gx > 1),
                        (gy % 1 < g) and (gy > 1),
                        (gxi % 1 < g) and (gxi > 1),
                        (gyi % 1 < g) and (gyi > 1)]
                for d in range(5):
                    if not dirs[d]:
                        continue
                    ox, oy = offs[d]
                    gij_x = int(gx - ox)
                    gij_y = int(gy - oy)
                    gj = min(max(gij_y, 0), int(H) - 1)
                    gi = min(max(gij_x, 0), int(W) - 1)
                    rows.append({
                        "expanded": [a, t[0], gx, gy, gwh[0], gwh[1], d],
                        "index": [a, gj, gi],
                        "tbox": [gx - gij_x, gy - gij_y, gwh[0], gwh[1]],
                        "anch": list(anchors[i][a]),
                    })
        layers.append({"gain": [1.0, 1.0, W, H, W, H, 1.0],
                       "wh_keep": keep, "rows": rows})
    return layers


def _close(a, b, tol=1e-9):
    assert math.isclose(a, b, rel_tol=tol, abs_tol=tol), f"{a} != {b}"


def _compare(solve, targets, anchors, shapes, anchor_t=4.0):
    led = solve(targets, anchors, shapes, anchor_t)
    ref = _oracle(targets, anchors, shapes, anchor_t)
    assert len(led) == len(ref)
    for l_led, l_ref in zip(led, ref):
        assert l_led["gain"] == l_ref["gain"]
        assert l_led["wh_keep"] == l_ref["wh_keep"]
        assert len(l_led["expanded"]) == len(l_ref["rows"])
        for e_led, i_led, r_ref in zip(l_led["expanded"], l_led["indices"],
                                       l_ref["rows"]):
            assert e_led[:2] == r_ref["expanded"][:2]
            assert e_led[6] == r_ref["expanded"][6]
            for a, b in zip(e_led[2:6], r_ref["expanded"][2:6]):
                _close(a, b)
            assert i_led == r_ref["index"], (i_led, r_ref["index"])
            j = l_led["expanded"].index(e_led)
            for a, b in zip(l_led["tbox"][j], r_ref["tbox"]):
                _close(a, b)
            assert l_led["anch"][j] == r_ref["anch"]
    return led


def check_basic_single_gt(solve):
    # 单 GT：32x32 图、格空间 x=y=16.0、w=h=0.5→16.0
    # anchor8: r=2 (1/r=0.5) keep；anchor16: r=1 keep；anchor32: 1/r=2 keep
    anchors = [[[8, 8], [16, 16], [32, 32]]]
    shapes = [[32, 32]]
    led = _compare(solve, [[0, 0.5, 0.5, 0.5, 0.5]], anchors, shapes)
    assert led[0]["wh_keep"] == [True] * 3
    # x=y=16.0 整数格：j/k/l/m 全真（%1=0<0.5 且坐标与逆坐标 >1）
    # 每 anchor 五副本，共 15 行
    exp = led[0]["expanded"]
    assert len(exp) == 15
    assert sorted((e[0], e[6]) for e in exp) == \
        [(a, d) for a in (0, 1, 2) for d in range(5)]
    # gij：dir0→16；dir1(+0.5x)→15；dir2(+0.5y)→15；dir3(−0.5x)→16；dir4→16
    for a in (0, 1, 2):
        rows = {e[6]: i for e, i in zip(exp, led[0]["indices"])
                if e[0] == a}
        assert rows[0] == [a, 16, 16] and rows[1] == [a, 16, 15]
        assert rows[2] == [a, 15, 16] and rows[3] == [a, 16, 16]
        assert rows[4] == [a, 16, 16]
    # tbox 残差：dir0 → 0.0；dir1 → gx−gij = 16−15 = 1.0
    tb0 = led[0]["tbox"][0]
    _close(tb0[0], 0.0)
    tb1 = [t for e, t in zip(exp, led[0]["tbox"]) if e[6] == 1][0]
    _close(tb1[0], 1.0)
    tb2 = [t for e, t in zip(exp, led[0]["tbox"]) if e[6] == 2][0]
    _close(tb2[1], 1.0)


def check_wh_ratio_filter(solve):
    # 双向比：max(r, 1/r) < anchor_t；小 GT 配大 anchor 由 1/r 过滤
    anchors = [[[8, 8], [16, 16], [32, 32]]]
    shapes = [[32, 32]]
    # w=0.3 → 格 9.6：r(8)=1.2 keep；r(16)=0.6→1/r=1.667 keep；r(32)=0.3→3.33
    # anchor_t=2.0：32 过滤
    led = _compare(solve, [[0, 0.5, 0.5, 0.3, 0.3]], anchors, shapes,
                   anchor_t=2.0)
    assert led[0]["wh_keep"] == [True, True, False]
    # anchor_t=1.5：16 (1.667) 与 32 都过滤
    led = _compare(solve, [[0, 0.5, 0.5, 0.3, 0.3]], anchors, shapes,
                   anchor_t=1.5)
    assert led[0]["wh_keep"] == [True, False, False]
    # 小 GT w=0.2 → 格 6.4：r(32)=0.2→1/r=5 ≥4 过滤（默认 4.0）
    led = _compare(solve, [[0, 0.5, 0.5, 0.2, 0.2]], anchors, shapes)
    assert led[0]["wh_keep"] == [True, True, False]


def check_offset_edge_guards(solve):
    # 双门：%1<0.5 且 坐标（或 W−x）>1；靠边不扩展
    anchors = [[[8, 8]]]
    shapes = [[16, 16]]
    # x=15.7：%1=0.7 → j False；gxi=0.3 → l False → 只 dir0
    led = _compare(solve, [[0, 15.7 / 16, 8.5 / 16, 0.2, 0.2]], anchors, shapes)
    assert [e[6] for e in led[0]["expanded"]] == [0]
    # x=14.6：%1=0.6 → j False；gxi=1.4>1 且 %1=0.4 → l True → [0,3]
    led = _compare(solve, [[0, 14.6 / 16, 8.5 / 16, 0.2, 0.2]], anchors, shapes)
    assert sorted(e[6] for e in led[0]["expanded"]) == [0, 3]
    idx = [i[2] for i in led[0]["indices"]]
    assert idx == [14, 15]        # dir0 → int(14.6)=14；dir3 → int(15.1)=15
    # x=1.2：gxy>1 且 %1=0.2 → j True；gxi=14.8 %1=0.8 → l False → [0,1]
    led = _compare(solve, [[0, 1.2 / 16, 8.5 / 16, 0.2, 0.2]], anchors, shapes)
    assert sorted(e[6] for e in led[0]["expanded"]) == [0, 1]
    assert [i[2] for i in led[0]["indices"]] == [1, 0]


def check_offset_fraction_gate(solve):
    # 小数部分恰 0.5 不扩展；对侧由 gxi 小数决定
    anchors = [[[8, 8]]]
    shapes = [[16, 16]]
    # x=13.7：%1=0.7 → j False；gxi=2.3 %1=0.3 → l True → [0,3]
    led = _compare(solve, [[0, 13.7 / 16, 8.5 / 16, 0.2, 0.2]], anchors, shapes)
    assert sorted(e[6] for e in led[0]["expanded"]) == [0, 3]
    # x=4.5：%1=0.5 不 <0.5 → j False；gxi=11.5 %1=0.5 → l False → [0]
    led = _compare(solve, [[0, 4.5 / 16, 8.5 / 16, 0.2, 0.2]], anchors, shapes)
    assert [e[6] for e in led[0]["expanded"]] == [0]


def check_clamp_defensive_only(solve):
    # 官方双门保证扩展不越界（gx>1 / gxi>1），clamp 是防御分支：
    # 正常输入下永不触发——随机对拍覆盖；此处只锁 1x1 图无正样本
    # （gw ≤1 → 1/r ≥ 8 恒被过滤）
    anchors = [[[8, 8]]]
    shapes = [[1, 1]]
    led = _compare(solve, [[0, 0.9, 0.9, 0.5, 0.5]], anchors, shapes)
    assert led[0]["wh_keep"] == [False]
    assert led[0]["expanded"] == []


def check_boundary_and_nonsquare(solve):
    # 边界 1：max(r,1/r) 恰等于 anchor_t → 过滤（严格 <）
    anchors = [[[8, 8]]]
    shapes = [[32, 32]]
    led = _compare(solve, [[0, 0.5, 0.5, 1.0, 1.0]], anchors, shapes)
    # 格宽 32、r=4.0 恰等 → False
    assert led[0]["wh_keep"] == [False]
    # 边界 2：非方形 gain（W≠H）——x 用 W、y 用 H；gxi=W−x
    anchors2 = [[[8, 8]]]
    shapes2 = [[8, 16]]          # H=8, W=16
    # gx=15.2（x=0.95）：j True（%1=0.2 & >1）、l False（gxi=0.8）
    # gy=7.6（y=0.95）：k False（%1=0.6）、m False（gyi=0.4）
    led = _compare(solve, [[0, 0.95, 0.95, 0.2, 0.4]], anchors2, shapes2)
    assert sorted(e[6] for e in led[0]["expanded"]) == [0, 1]
    # 边界 3：坐标恰 =1.0 → >1 False 不扩展（>=1 变异会扩展）
    # gx=1.0（x=1/16）、gy=4.5（y=0.5625）：j False（恰 1 不>1）；
    # gxi=15 %1=0 → l True；gy/gyi %1=0.5 → k/m False
    led = _compare(solve, [[0, 1.0 / 16, 0.5625, 0.2, 0.4]],
                   anchors2, shapes2)
    assert sorted(e[6] for e in led[0]["expanded"]) == [0, 3]
    # x 恰整数：dir0 与 dir3 都落 int(1.0)=1 同格（quirk，tbox 残差不同）
    assert [i[2] for i in led[0]["indices"]] == [1, 1]


def check_multi_layer_anchor_stride(solve):
    # 官方 r = 格空间 wh / 像素 anchor 的混搭：同一 GT 跨层 r 逐层 ×1/4
    # （格宽 ÷2、anchor ×2）→ 只有比值落 (1/anchor_t, anchor_t) 的层 keep
    anchors = [[[4, 4]], [[8, 8]], [[16, 16]]]
    shapes = [[8, 8], [4, 4], [2, 2]]
    # w=0.6：层0 格 4.8 vs 4 → r=1.2 keep；层1 格 2.4 vs 8 → 1/r=3.33 keep；
    # 层2 格 1.2 vs 16 → 1/r=13.3 过滤
    led = _compare(solve, [[0, 0.5, 0.5, 0.6, 0.6]], anchors, shapes)
    assert [l["wh_keep"][0] for l in led] == [True, True, False]


def check_no_targets(solve):
    # nt=0：官方 t=targets[0]、offsets=0 → 全层空
    anchors = [[[8, 8]]]
    shapes = [[8, 8]]
    led = _compare(solve, [], anchors, shapes)
    for l in led:
        assert l["expanded"] == [] and l["indices"] == []
        assert l["tbox"] == [] and l["anch"] == []


def check_random_against_oracle(solve):
    rng = random.Random(707)
    for _ in range(200):
        nt = rng.randrange(0, 5)
        targets = [[rng.randrange(3),
                    round(rng.uniform(0.05, 0.95), 2),
                    round(rng.uniform(0.05, 0.95), 2),
                    round(rng.uniform(0.01, 0.5), 3),
                    round(rng.uniform(0.01, 0.5), 3)]
                   for _ in range(nt)]
        anchors = [[[8, 8], [16, 16], [32, 32]],
                   [[16, 16], [32, 32], [64, 64]],
                   [[32, 32], [64, 64], [128, 128]]]
        shapes = [[8, 8], [4, 4], [2, 2]]
        anchor_t = rng.choice([1.0, 2.0, 4.0, 8.0])
        _compare(solve, targets, anchors, shapes, anchor_t)


def check_invalid_inputs(solve):
    good = dict(targets=[[0, 0.5, 0.5, 0.1, 0.1]],
                anchors=[[[8, 8], [16, 16], [32, 32]]],
                shapes=[[8, 8]], anchor_t=4.0)

    def call(**kw):
        a = {**good, **kw}
        return solve(a["targets"], a["anchors"], a["shapes"], a["anchor_t"])
    for kw in [
        dict(anchors=[]),
        dict(shapes=[]),
        dict(shapes=[[8]]),                          # shape 非 (H,W)
        dict(shapes=[[0, 8]]),
        dict(anchors=[[[0, 8], [16, 16], [32, 32]]]),  # anchor w=0
        dict(anchors=[[[8], [16, 16], [32, 32]]]),   # anchor 单值
        dict(anchor_t=0.0),
        dict(anchor_t=-1.0),
        dict(targets=[[0, 0.5, 0.5, 0.1]]),          # 行长 4
        dict(targets=[[0, 1.2, 0.5, 0.1, 0.1]]),     # x 越界
        dict(targets=[[0, 0.5, 0.5, 0.0, 0.1]]),     # w=0
        dict(targets=[[0, 0.5, 0.5, 1.1, 0.1]]),     # w>1
        dict(targets=[[-1, 0.5, 0.5, 0.1, 0.1]]),    # 类负
        dict(targets=[[0.5, 0.5, 0.5, 0.1, 0.1]]),   # 类非 int
        dict(targets=[[0, float("nan"), 0.5, 0.1, 0.1]]),
    ]:
        try:
            call(**kw)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError(f"expected error: {kw}")
