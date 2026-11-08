"""845 Captum Occlusion 滑动窗口扰动归因账 —— 判题用例。

四个 check 函数，均以 solve 为唯一入参（judge 内部 fn(solve) 调用）。
solve(mode, **kw) 分派：
  occlusion_grid / occlusion_mask / occlusion_diff / occlusion_attribution
"""

import math

TOL = 1e-12
REL_TOL = 1e-9


def _close(a, b, tol=TOL):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def _cmp_scalar(got, want, label, tol=TOL):
    assert isinstance(got, (int, float)) and not isinstance(got, bool), (
        "%s: expected number, got %r" % (label, type(got))
    )
    assert _close(got, want, tol), "%s: got %r want %r" % (label, got, want)


def _cmp_list(got, want, label, tol=TOL):
    assert isinstance(got, list), "%s: expected list, got %r" % (label, type(got))
    assert len(got) == len(want), (
        "%s: length mismatch got %d want %d" % (label, len(got), len(want))
    )
    for i, (g, w) in enumerate(zip(got, want)):
        assert isinstance(g, (int, float)), "%s[%d]: expected number" % (label, i)
        assert _close(g, w, tol), "%s[%d]: got %r want %r" % (label, i, g, w)


def _cmp_int_list(got, want, label):
    assert isinstance(got, list), "%s: expected list" % label
    assert len(got) == len(want), "%s: length mismatch" % label
    for i, (g, w) in enumerate(zip(got, want)):
        assert isinstance(g, int) and not isinstance(g, bool), (
            "%s[%d]: expected int, got %r" % (label, i, g)
        )
        assert g == w, "%s[%d]: got %r want %r" % (label, i, g, w)


def _cmp_nested(got, want, label):
    assert isinstance(got, list), "%s: expected list" % label
    assert len(got) == len(want), "%s: length mismatch" % label
    for i, (g, w) in enumerate(zip(got, want)):
        if isinstance(w, list):
            _cmp_nested(g, w, "%s[%d]" % (label, i))
        else:
            assert isinstance(g, int) and not isinstance(g, bool), (
                "%s[%d]: expected int, got %r" % (label, i, g)
            )
            assert g == w, "%s[%d]: got %r want %r" % (label, i, g, w)


def _expect_error(fn, label, exc=ValueError):
    try:
        fn()
    except exc:
        return
    except Exception as e:  # noqa: BLE001
        raise AssertionError(
            "%s: expected %s, got %s: %s" % (label, exc.__name__, type(e).__name__, e)
        )
    raise AssertionError("%s: expected %s, but no error raised" % (label, exc.__name__))


# ---------------------------------------------------------------------------
# 官方语义常量（用于锚点断言，均已与官方 Captum 0.9.0 逐位对拍确认）
# ---------------------------------------------------------------------------

# L=5, W=3, S=1 → shift_count = ceil((5-3)/1)+1 = 3，偏移 0/1/2，窗口不截断
GRID_5_3_1 = {
    "shift_counts": [3],
    "num_windows": 3,
    "offsets": [[0], [1], [2]],
    "extents": [[[0, 3]], [[1, 4]], [[2, 5]]],
}

# L=6, W=4, S=3 → shift_count = ceil((6-4)/3)+1 = 2，偏移 0/3；
# 第二个窗口 offset=3，3+4=7 > 6 → 官方 F.pad 右侧负 padding → 窗口被截断成 [3,6)
GRID_6_4_3 = {
    "shift_counts": [2],
    "num_windows": 2,
    "offsets": [[0], [3]],
    "extents": [[[0, 4]], [[3, 6]]],
}

# L=10, W=3, S=3 → shift_count = ceil((10-3)/3)+1 = 4，偏移 0/3/6/9；
# 末窗 offset=9，9+3=12 > 10 → 截断成 [9,10)
GRID_10_3_3 = {
    "shift_counts": [4],
    "num_windows": 4,
    "offsets": [[0], [3], [6], [9]],
    "extents": [[[0, 3]], [[3, 6]], [[6, 9]], [[9, 10]]],
}

# L=3, W=4 非法（窗口超过输入维）
# L=5, W=3, S=4 非法（stride > window 且窗口未覆盖整维）

# 覆盖计数：L=5, W=3, S=1 → 中间位置被 3 个窗口覆盖
COVERAGE_5_3_1 = [1, 2, 3, 2, 1]

# 覆盖计数：L=6, W=4, S=3 → 位置 3 被两个窗口覆盖
COVERAGE_6_4_3 = [1, 1, 1, 2, 1, 1]

# 前向：A5 = 线性层 W=[[1,2,3,4,5]], b=[0.5]，f(x) = sum(x_i * (i+1)) + 0.5
LAYER_A5 = [{"W": [[1.0, 2.0, 3.0, 4.0, 5.0]], "b": [0.5]}]
X5 = [1.0, 2.0, 3.0, 4.0, 5.0]

# 输入维为 2 的层：用于盯住「x 长度 vs shape 大小」这条独立防线
LAYER_2IN = [{"W": [[1.0, 1.0]], "b": [0.0]}]

# 前向：A5b = 两个输出
LAYER_A5B = [
    {"W": [[1.0, 2.0, 3.0, 4.0, 5.0], [-1.0, 0.5, 2.0, 0.0, 1.0]],
     "b": [0.5, -0.25]}
]


def check_occlusion_grid(solve):
    # 基础 1-D
    r = solve("occlusion_grid", shape=[5], window_shape=[3], strides=[1])
    _cmp_int_list(r["shift_counts"], GRID_5_3_1["shift_counts"], "grid.sc")
    assert r["num_windows"] == 3, "grid.num_windows"
    _cmp_nested(r["offsets"], GRID_5_3_1["offsets"], "grid.offsets")
    for i, e in enumerate(GRID_5_3_1["extents"]):
        _cmp_nested(r["extents"][i], e, "grid.extents[%d]" % i)

    # ★ 标量 stride 广播：官方 _format_and_verify_strides 允许 strides 是
    #   一个标量 int（对单输入张量包成 (strides,)），适用于所有维度。
    rs = solve("occlusion_grid", shape=[5], window_shape=[3], strides=2)
    _cmp_int_list(rs["shift_counts"], [2], "grid.scalar.sc")
    _cmp_nested(rs["offsets"], [[0], [2]], "grid.scalar.offsets")

    # ★ 标量 stride 广播在 2-D 下必须各维都生效
    #   （1-D 下「只改第一维」与「全部广播」等价，1-D 用例抓不住）
    #   shape=[3,4], window=[2,2], strides=2（标量）
    #   dim0: ceil((3-2)/2)+1 = 2；dim1: ceil((4-2)/2)+1 = 2
    rs2 = solve("occlusion_grid", shape=[3, 4], window_shape=[2, 2], strides=2)
    _cmp_int_list(rs2["shift_counts"], [2, 2], "grid.scalar2d.sc")
    _cmp_nested(
        rs2["offsets"], [[0, 0], [2, 0], [0, 2], [2, 2]], "grid.scalar2d.offsets"
    )

    # strides=None → 默认全 1
    r2 = solve("occlusion_grid", shape=[5], window_shape=[3])
    _cmp_int_list(r2["shift_counts"], [3], "grid.def.sc")
    _cmp_nested(r2["offsets"], [[0], [1], [2]], "grid.def.offsets")

    # ★ 负 padding 截断：L=6, W=4, S=3
    r3 = solve("occlusion_grid", shape=[6], window_shape=[4], strides=[3])
    _cmp_int_list(r3["shift_counts"], GRID_6_4_3["shift_counts"], "grid.tr.sc")
    _cmp_nested(r3["offsets"], GRID_6_4_3["offsets"], "grid.tr.offsets")
    _cmp_nested(r3["extents"][1], [[3, 6]], "grid.tr.extents[1]")

    # ★ 末窗截断：L=10, W=3, S=3
    r4 = solve("occlusion_grid", shape=[10], window_shape=[3], strides=[3])
    _cmp_int_list(r4["shift_counts"], GRID_10_3_3["shift_counts"], "grid.tr2.sc")
    _cmp_nested(r4["offsets"], GRID_10_3_3["offsets"], "grid.tr2.offsets")
    _cmp_nested(r4["extents"][3], [[9, 10]], "grid.tr2.extents[3]")

    # 窗口恰好覆盖整维 → 只有 1 个窗口
    r5 = solve("occlusion_grid", shape=[5], window_shape=[5], strides=[1])
    _cmp_int_list(r5["shift_counts"], [1], "grid.full.sc")
    assert r5["num_windows"] == 1, "grid.full.num_windows"
    _cmp_nested(r5["offsets"], [[0]], "grid.full.offsets")

    # ★ base-conversion 枚举顺序：dim0 是最低位（变化最快）
    # H=3, W=4, win=(2,2), stride=(1,2) → shift_counts=(2,2)
    # 官方顺序：(0,0) (1,0) (0,2) (1,2)
    r6 = solve("occlusion_grid", shape=[3, 4], window_shape=[2, 2], strides=[1, 2])
    _cmp_int_list(r6["shift_counts"], [2, 2], "grid2d.sc")
    _cmp_nested(
        r6["offsets"], [[0, 0], [1, 0], [0, 2], [1, 2]], "grid2d.offsets"
    )

    # 3-D
    r7 = solve("occlusion_grid", shape=[2, 3, 2], window_shape=[1, 2, 1],
                strides=[1, 1, 1])
    _cmp_int_list(r7["shift_counts"], [2, 2, 2], "grid3d.sc")
    assert r7["num_windows"] == 8, "grid3d.num_windows"
    _cmp_nested(r7["offsets"][0], [0, 0, 0], "grid3d.offsets[0]")
    _cmp_nested(r7["offsets"][1], [1, 0, 0], "grid3d.offsets[1]")
    _cmp_nested(r7["offsets"][2], [0, 1, 0], "grid3d.offsets[2]")

    # 非法：窗口超过输入维
    _expect_error(
        lambda: solve("occlusion_grid", shape=[5], window_shape=[7], strides=[1]),
        "grid.window>input",
    )
    # 非法：stride > window 且窗口未覆盖整维
    _expect_error(
        lambda: solve("occlusion_grid", shape=[5], window_shape=[3], strides=[4]),
        "grid.stride>window",
    )
    # ★ 窗口恰好覆盖整维时 stride 可以大于 window
    r8 = solve("occlusion_grid", shape=[5], window_shape=[5], strides=[7])
    _cmp_int_list(r8["shift_counts"], [1], "grid.fullbigstride.sc")
    assert r8["num_windows"] == 1, "grid.fullbigstride.num_windows"
    # 非法：窗口维度数与输入维度数不一致
    _expect_error(
        lambda: solve("occlusion_grid", shape=[3, 4], window_shape=[2], strides=[1, 1]),
        "grid.window-arity",
    )
    # 非法：stride 维度数不一致
    _expect_error(
        lambda: solve("occlusion_grid", shape=[3, 4], window_shape=[2, 2], strides=[1]),
        "grid.stride-arity",
    )
    # 非法：窗口恰好比输入大 1（边界，不能只拦得住「大 2 以上」）
    _expect_error(
        lambda: solve("occlusion_grid", shape=[5], window_shape=[6], strides=[1]),
        "grid.window-exceed-by-one",
    )
    _expect_error(
        lambda: solve("occlusion_mask", shape=[5], window_shape=[6], strides=[1]),
        "mask.window-exceed-by-one",
    )
    # 非法：非正维度
    _expect_error(
        lambda: solve("occlusion_grid", shape=[5, 0], window_shape=[1, 1]),
        "grid.nonpositive",
    )


def check_occlusion_mask(solve):
    # 覆盖计数
    r = solve("occlusion_mask", shape=[5], window_shape=[3], strides=[1])
    _cmp_int_list(r["coverage"], COVERAGE_5_3_1, "mask.coverage")
    assert len(r["masks"]) == 3, "mask.num_masks"
    _cmp_int_list(r["masks"][0], [1, 1, 1, 0, 0], "mask.masks[0]")
    _cmp_int_list(r["masks"][1], [0, 1, 1, 1, 0], "mask.masks[1]")
    _cmp_int_list(r["masks"][2], [0, 0, 1, 1, 1], "mask.masks[2]")

    # ★ 截断窗口的掩码（长度只有 3，不是 4）
    r2 = solve("occlusion_mask", shape=[6], window_shape=[4], strides=[3])
    _cmp_int_list(r2["coverage"], COVERAGE_6_4_3, "mask.tr.coverage")
    assert len(r2["masks"]) == 2, "mask.tr.num_masks"
    _cmp_int_list(r2["masks"][0], [1, 1, 1, 1, 0, 0], "mask.tr.masks[0]")
    _cmp_int_list(r2["masks"][1], [0, 0, 0, 1, 1, 1], "mask.tr.masks[1]")

    # ★ 末窗截断：L=10, W=3, S=3 → 最后一个掩码只有 1 个 1
    r3 = solve("occlusion_mask", shape=[10], window_shape=[3], strides=[3])
    assert len(r3["masks"]) == 4, "mask.tr2.num_masks"
    _cmp_int_list(r3["masks"][3], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], "mask.tr2.masks[3]")
    _cmp_int_list(r3["coverage"], [1] * 10, "mask.tr2.coverage")

    # ★ 窗口恰好覆盖整维 → 只有一个掩码，覆盖全部
    r4 = solve("occlusion_mask", shape=[5], window_shape=[5], strides=[1])
    assert len(r4["masks"]) == 1, "mask.full.num_masks"
    _cmp_int_list(r4["masks"][0], [1] * 5, "mask.full.masks[0]")
    _cmp_int_list(r4["coverage"], [1] * 5, "mask.full.coverage")

    # 2-D：row-major 展开
    r5 = solve("occlusion_mask", shape=[3, 4], window_shape=[2, 2], strides=[1, 1])
    assert len(r5["masks"]) == 6, "mask2d.num_masks"
    # 第一个窗口在左上角，覆盖 (0,0),(0,1),(1,0),(1,1) → row-major 下标 0,1,4,5
    _cmp_int_list(
        r5["masks"][0], [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0], "mask2d.masks[0]"
    )
    # 第二个窗口 dim0 偏移 1（低位先变）→ 覆盖 (1,0),(1,1),(2,0),(2,1)
    _cmp_int_list(
        r5["masks"][1], [0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0], "mask2d.masks[1]"
    )
    # 第三个窗口 dim1 偏移 1 → 覆盖 (0,1),(0,2),(1,1),(1,2)
    _cmp_int_list(
        r5["masks"][2], [0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0], "mask2d.masks[2]"
    )
    # 覆盖计数：中间 2x2 区域被 4 个窗口覆盖
    _cmp_int_list(
        r5["coverage"], [1, 2, 2, 1, 2, 4, 4, 2, 1, 2, 2, 1], "mask2d.coverage"
    )

    # ★ 多维窗口截断：shape=[3,4], window=[2,3], stride=[1,2]
    #   dim1: shift_count = ceil((4-3)/2)+1 = 2 → dim1 偏移 0/2
    #   偏移 2 时 2+3 = 5 > 4 → 该维窗口被截断成 [2,4)
    r6 = solve("occlusion_mask", shape=[3, 4], window_shape=[2, 3], strides=[1, 2])
    assert len(r6["masks"]) == 4, "mask2dtr.num_masks"
    # 窗口 0：dim0 偏移 0、dim1 偏移 0 → 覆盖 (0,0),(0,1),(0,2),(1,0),(1,1),(1,2)
    _cmp_int_list(
        r6["masks"][0],
        [1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        "mask2dtr.masks[0]",
    )
    # 窗口 2：dim0 偏移 0、dim1 偏移 2 → dim1 只覆盖列 2、3（列 3 被截断掉？不，列 2..4 截断为 2..3）
    #   dim1 偏移 2、窗口宽 3 → 覆盖列 2,3,4 但输入只有 4 列 → 截断为列 2,3
    _cmp_int_list(
        r6["masks"][2],
        [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0],
        "mask2dtr.masks[2]",
    )
    # 窗口 3：dim0 偏移 1、dim1 偏移 2
    _cmp_int_list(
        r6["masks"][3],
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
        "mask2dtr.masks[3]",
    )


def check_occlusion_diff(solve):
    # A5 前向：f(x) = 1*x0 + 2*x1 + 3*x2 + 4*x3 + 5*x4 + 0.5
    # x = [1,2,3,4,5] → f = 1+4+9+16+25+0.5 = 55.5
    r = solve(
        "occlusion_diff", layers=LAYER_A5, x=X5, baseline=0.0,
        shape=[5], window_shape=[3], strides=[1], target=0,
    )
    _cmp_scalar(r["initial_eval"], 55.5, "diff.initial")
    assert len(r["window_evals"]) == 3, "diff.num_window_evals"
    assert len(r["eval_diffs"]) == 3, "diff.num_eval_diffs"
    # 窗口 0 遮挡 [0,3)：f 少算 1+4+9 = 14 → 41.5，diff = 14
    _cmp_scalar(r["window_evals"][0], 41.5, "diff.we[0]")
    _cmp_scalar(r["eval_diffs"][0], 14.0, "diff.ed[0]")
    # 窗口 1 遮挡 [1,4)：少算 4+9+16 = 29 → 26.5
    _cmp_scalar(r["window_evals"][1], 26.5, "diff.we[1]")
    _cmp_scalar(r["eval_diffs"][1], 29.0, "diff.ed[1]")
    # 窗口 2 遮挡 [2,5)：少算 9+16+25 = 50 → 5.5
    _cmp_scalar(r["window_evals"][2], 5.5, "diff.we[2]")
    _cmp_scalar(r["eval_diffs"][2], 50.0, "diff.ed[2]")
    # 未归一化累加：位置 0 只被窗口 0 覆盖 → 14
    # 位置 1 被窗口 0,1 覆盖 → 14+29 = 43
    # 位置 2 被 0,1,2 覆盖 → 14+29+50 = 93
    # 位置 3 被 1,2 → 29+50 = 79；位置 4 被 2 → 50
    _cmp_list(r["total_attrib"], [14.0, 43.0, 93.0, 79.0, 50.0], "diff.total_attrib")

    # ★ baseline 非零：baseline=10 时遮挡位置被替换成 10
    r2 = solve(
        "occlusion_diff", layers=LAYER_A5, x=X5, baseline=10.0,
        shape=[5], window_shape=[3], strides=[1], target=0,
    )
    # 窗口 0：[0,3) 被替换成 10 → f = 10+20+30+16+25+0.5 = 101.5
    # diff = 55.5 - 101.5 = -46
    _cmp_scalar(r2["window_evals"][0], 101.5, "diff.bl.we[0]")
    _cmp_scalar(r2["eval_diffs"][0], -46.0, "diff.bl.ed[0]")

    # ★ 多输出选 target=1
    r3 = solve(
        "occlusion_diff", layers=LAYER_A5B, x=X5, baseline=0.0,
        shape=[5], window_shape=[3], strides=[1], target=1,
    )
    # f1 = -1*1 + 0.5*2 + 2*3 + 0*4 + 1*5 - 0.25 = -1+1+6+0+5-0.25 = 10.75
    _cmp_scalar(r3["initial_eval"], 10.75, "diff.t1.initial")
    # 窗口 0 遮挡 [0,3)：-1+1+6 被抹掉 → 0+0+0+0+5-0.25 = 4.75 → diff = 6
    _cmp_scalar(r3["eval_diffs"][0], 6.0, "diff.t1.ed[0]")

    # ★ 截断窗口的差分
    r4 = solve(
        "occlusion_diff", layers=[{"W": [[1.0] * 6], "b": [0.0]}],
        x=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], baseline=0.0,
        shape=[6], window_shape=[4], strides=[3], target=0,
    )
    # 初始 = 21；窗口 0 遮挡 [0,4) → 剩 5+6 = 11，diff = 10
    _cmp_scalar(r4["initial_eval"], 21.0, "diff.tr.initial")
    _cmp_scalar(r4["eval_diffs"][0], 10.0, "diff.tr.ed[0]")
    # 窗口 1 遮挡 [3,6)（截断成 3 个位置）→ 剩 1+2+3 = 6，diff = 15
    _cmp_scalar(r4["eval_diffs"][1], 15.0, "diff.tr.ed[1]")

    # ★ 末层输出为负 → ReLU 是否作用在最后一层必须可观测
    #   NEG: f(x) = x0+2x1+3x2+4x3+5x4 - 100；x=[1,2,3,4,5] → 55-100 = -45
    #   窗口 0 遮挡 [0,3) → 0+0+0+16+25-100 = -59 → diff = -45-(-59) = 14
    neg = [{"W": [[1.0, 2.0, 3.0, 4.0, 5.0]], "b": [-100.0]}]
    rneg = solve(
        "occlusion_diff", layers=neg, x=X5, baseline=0.0,
        shape=[5], window_shape=[3], strides=[1], target=0,
    )
    _cmp_scalar(rneg["initial_eval"], -45.0, "diff.neg.initial")
    _cmp_scalar(rneg["window_evals"][0], -59.0, "diff.neg.we[0]")
    _cmp_scalar(rneg["eval_diffs"][0], 14.0, "diff.neg.ed[0]")

    # ★ 中间层的 ReLU 必须可观测（去掉 ReLU 或改变 ReLU 作用位置都要被抓到）
    #   层0: W=[[1,-1]], b=[0]；层1: W=[[1]], b=[0]
    #   x=[1,2] → 层0 输出 -1（被 ReLU 截为 0）→ 层1 输出 0
    #   窗口 0 遮挡位置 0 → x=[0,2] → 层0 输出 -2 → ReLU 截为 0 → 输出仍 0，diff = 0
    #   窗口 1 遮挡位置 1 → x=[1,0] → 层0 输出 1 → 输出 1，diff = -1
    mid_relu = [
        {"W": [[1.0, -1.0]], "b": [0.0]},
        {"W": [[1.0]], "b": [0.0]},
    ]
    rmid = solve(
        "occlusion_diff", layers=mid_relu, x=[1.0, 2.0], baseline=0.0,
        shape=[2], window_shape=[1], strides=[1], target=0,
    )
    _cmp_scalar(rmid["initial_eval"], 0.0, "diff.relu.initial")
    _cmp_scalar(rmid["window_evals"][0], 0.0, "diff.relu.we[0]")
    _cmp_scalar(rmid["window_evals"][1], 1.0, "diff.relu.we[1]")
    _cmp_scalar(rmid["eval_diffs"][0], 0.0, "diff.relu.ed[0]")
    _cmp_scalar(rmid["eval_diffs"][1], -1.0, "diff.relu.ed[1]")

    # ★ 负 target 索引：官方走 _verify_select_column → output[(slice(None), -1)]
    #   即 PyTorch 负索引语义：target=-1 选最后一个输出，target=-2 选倒数第二个。
    #   LAYER_A5B 有 2 个输出 → target=-1 等价 target=1，target=-2 等价 target=0。
    rneg = solve(
        "occlusion_diff", layers=LAYER_A5B, x=X5, baseline=0.0,
        shape=[5], window_shape=[3], strides=[1], target=-1,
    )
    rpos = solve(
        "occlusion_diff", layers=LAYER_A5B, x=X5, baseline=0.0,
        shape=[5], window_shape=[3], strides=[1], target=1,
    )
    _cmp_scalar(rneg["initial_eval"], rpos["initial_eval"], "diff.neg1.initial")
    assert len(rneg["eval_diffs"]) == len(rpos["eval_diffs"]), "diff.neg1.len"
    for i, (a, b) in enumerate(zip(rneg["eval_diffs"], rpos["eval_diffs"])):
        _cmp_scalar(a, b, "diff.neg1.ed[%d]" % i)
    rneg2 = solve(
        "occlusion_diff", layers=LAYER_A5B, x=X5, baseline=0.0,
        shape=[5], window_shape=[3], strides=[1], target=-2,
    )
    rpos0 = solve(
        "occlusion_diff", layers=LAYER_A5B, x=X5, baseline=0.0,
        shape=[5], window_shape=[3], strides=[1], target=0,
    )
    _cmp_scalar(rneg2["initial_eval"], rpos0["initial_eval"], "diff.neg2.initial")

    # 非法：target 越界（正向）
    _expect_error(
        lambda: solve(
            "occlusion_diff", layers=LAYER_A5, x=X5, baseline=0.0,
            shape=[5], window_shape=[3], strides=[1], target=3,
        ),
        "diff.target-oob",
    )
    # 非法：target 负向越界（-3 对 1 个输出）
    _expect_error(
        lambda: solve(
            "occlusion_diff", layers=LAYER_A5, x=X5, baseline=0.0,
            shape=[5], window_shape=[3], strides=[1], target=-2,
        ),
        "diff.target-neg-oob",
    )
    # 非法：输入长度与 shape 不符
    # ★ 这里刻意用「层输入维 == x 长度 == 2」的层（LAYER_2IN）：
    #   若实现漏掉「x 长度必须等于 shape 大小」这道校验，
    #   _forward 的「层输入维」校验就不会兜住（两者不再指向同一个错误），
    #   从而这道用例能真正盯住「x 长度 vs shape 大小」这条独立防线。
    _expect_error(
        lambda: solve(
            "occlusion_diff", layers=LAYER_2IN, x=[1.0, 2.0], baseline=0.0,
            shape=[5], window_shape=[3], strides=[1], target=0,
        ),
        "diff.x-length",
    )
    # 非法：层输入维度与展平后大小不符
    _expect_error(
        lambda: solve(
            "occlusion_diff", layers=[{"W": [[1.0, 2.0]], "b": [0.0]}], x=X5,
            baseline=0.0, shape=[5], window_shape=[3], strides=[1], target=0,
        ),
        "diff.layer-dim",
    )
    # 非法：baseline 长度与 shape 不符
    _expect_error(
        lambda: solve(
            "occlusion_diff", layers=LAYER_A5, x=X5, baseline=[1.0, 2.0],
            shape=[5], window_shape=[3], strides=[1], target=0,
        ),
        "diff.baseline-length",
    )


def check_occlusion_attribution(solve):
    # ★ 核心：重叠区归因取平均（Occlusion 的 use_weights=True）
    r = solve(
        "occlusion_attribution", layers=LAYER_A5, x=X5, baseline=0.0,
        shape=[5], window_shape=[3], strides=[1], target=0,
    )
    _cmp_list(r["total_attrib"], [14.0, 43.0, 93.0, 79.0, 50.0], "attr.total_attrib")
    _cmp_int_list(r["weights"], [1, 2, 3, 2, 1], "attr.weights")
    # 归因 = total_attrib / weights = [14, 21.5, 31, 39.5, 50]
    _cmp_list(r["attribution"], [14.0, 21.5, 31.0, 39.5, 50.0], "attr.attribution")

    # ★ 关键判别：若错误地不做归一化（除以 weights），attribution 会等于 total_attrib
    #    第一个位置 weights=1，两者相同（14.0）；第二个位置必须不同（21.5 vs 43）
    assert abs(r["attribution"][1] - 43.0) > 1e-6, (
        "attr: 归因未按覆盖次数取平均（得到未归一化累加值）"
    )

    # ★ 窗口恰好覆盖整维 → 每个位置权重都是 1，归因 == total_attrib
    r2 = solve(
        "occlusion_attribution", layers=LAYER_A5, x=X5, baseline=0.0,
        shape=[5], window_shape=[5], strides=[1], target=0,
    )
    _cmp_int_list(r2["weights"], [1, 1, 1, 1, 1], "attr.full.weights")
    _cmp_list(r2["attribution"], [55.0, 55.0, 55.0, 55.0, 55.0], "attr.full.attribution")

    # ★ 截断场景：L=6, W=4, S=3
    r3 = solve(
        "occlusion_attribution", layers=[{"W": [[1.0] * 6], "b": [0.0]}],
        x=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], baseline=0.0,
        shape=[6], window_shape=[4], strides=[3], target=0,
    )
    _cmp_int_list(r3["weights"], [1, 1, 1, 2, 1, 1], "attr.tr.weights")
    # 窗口 0 diff=10 覆盖 [0,4)；窗口 1 diff=15 覆盖 [3,6)
    # 位置 3 被两个窗口覆盖 → (10+15)/2 = 12.5；其余为各自 diff
    _cmp_list(r3["total_attrib"], [10.0, 10.0, 10.0, 25.0, 15.0, 15.0], "attr.tr.total")
    _cmp_list(
        r3["attribution"], [10.0, 10.0, 10.0, 12.5, 15.0, 15.0], "attr.tr.attribution"
    )

    # ★ baseline 逐元素
    r4 = solve(
        "occlusion_attribution", layers=[{"W": [[1.0] * 5], "b": [0.0]}],
        x=X5, baseline=[10.0, 20.0, 30.0, 40.0, 50.0],
        shape=[5], window_shape=[3], strides=[1], target=0,
    )
    # 窗口 0：[0,3) 换成 [10,20,30] → sum = 60 + 4 + 5 = 69；初始 15 → diff = -54
    # 窗口 1：[1,4) 换成 [20,30,40] → 1 + 90 + 5 = 96 → diff = -81
    # 窗口 2：[2,5) 换成 [30,40,50] → 1 + 2 + 120 = 123 → diff = -108
    _cmp_list(
        r4["total_attrib"], [-54.0, -135.0, -243.0, -189.0, -108.0], "attr.bl.total"
    )
    _cmp_list(
        r4["attribution"], [-54.0, -67.5, -81.0, -94.5, -108.0], "attr.bl.attribution"
    )

    # ★ 2-D + 逐元素 baseline
    r5 = solve(
        "occlusion_attribution",
        layers=[
            {"W": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                   [-1.0, 0.5, 2.0, -3.0, 0.0, 1.0, -2.0, 3.0, 0.5, -1.0, 2.0, -0.5]],
             "b": [0.25, -0.75]},
            {"W": [[2.0, -1.0]], "b": [-0.5]},
        ],
        x=[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
        baseline=0.0, shape=[3, 4], window_shape=[2, 2], strides=[1, 2], target=0,
    )
    _cmp_int_list(
        r5["weights"], [1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1], "attr2d.weights"
    )
    assert len(r5["attribution"]) == 12, "attr2d.len"
    for v in r5["attribution"]:
        assert isinstance(v, float), "attr2d.value-type"

    # ★ 归因结果 = 1/3（不是二进制有限小数）→ float32 舍入必须可观测
    #   系数 c = [21, -10, 0, 5, -4]，x = [1,2,3,4,5]
    #   窗口 0 [0,3): 21*1 -10*2 + 0*3 = 1   → diff = 1
    #   窗口 1 [1,4): -10*2 + 0*3 + 5*4 = 0   → diff = 0
    #   窗口 2 [2,5): 0*3 + 5*4 - 4*5 = 0     → diff = 0
    #   位置 2 被 3 个窗口覆盖 → total = 1, weights = 3 → 归因 = 1/3
    third = [{"W": [[21.0, -10.0, 0.0, 5.0, -4.0]], "b": [0.0]}]
    rthird = solve(
        "occlusion_attribution", layers=third, x=X5, baseline=0.0,
        shape=[5], window_shape=[3], strides=[1], target=0,
    )
    _cmp_list(rthird["total_attrib"], [1.0, 1.0, 1.0, 0.0, 0.0], "attr.third.total")
    _cmp_int_list(rthird["weights"], [1, 2, 3, 2, 1], "attr.third.weights")
    # 归因 = float32(1/3) = 0.3333333432674407958984375
    # （官方的 format_result 里是 `single_attrib.float() / weight`，
    #   先降级到 float32 再在 float32 下做除法，故期望值必须按 float32 舍入给出）
    _cmp_list(
        rthird["attribution"],
        [1.0, 0.5, 0.3333333432674407958984375, 0.0, 0.0],
        "attr.third.attribution",
        tol=1e-12,
    )

    # 非法输入同样要报错
    _expect_error(
        lambda: solve(
            "occlusion_attribution", layers=LAYER_A5, x=X5, baseline=0.0,
            shape=[5], window_shape=[3], strides=[4], target=0,
        ),
        "attr.stride>window",
    )
