import torch

# 示例（实测回填）：两级 backbone（2x2 / 1x1），fpn_dim=2；第二级侧接带 bias
EXAMPLE_FEATURES = [
    [[[[1.0, 2.0], [3.0, 4.0]], [[10.0, 20.0], [30.0, 40.0]]]],
    [[[[5.0]], [[6.0]], [[7.0]]]],
]
EXAMPLE_WEIGHTS = [
    [[1.0, 0.0], [0.0, 1.0]],
    [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]],
]
EXAMPLE_BIASES = [None, [0.0, 1.0]]
EXAMPLE_FUSED = [
    [[[[12.0, 13.0], [14.0, 15.0]], [[24.0, 34.0], [44.0, 54.0]]]],
    [[[[11.0]], [[14.0]]]],
]
EXAMPLE_SIZES = [[2, 2], [1, 1]]
EXAMPLE_STRIDES = [4, 8]
EXAMPLE_LATERAL_PARAMS = [4, 8]
EXAMPLE_NUM_LATERAL_PARAMS = 12
EXAMPLE_NUM_UPSAMPLE = 1
EXAMPLE_ANCHORS_PER_LEVEL = [12, 3]
EXAMPLE_NUM_ANCHORS = 15


def _features(tensors, dtype=torch.float32):
    return [t.to(dtype) for t in tensors]


def _weights(lists, dtype=torch.float32):
    return [torch.tensor(w, dtype=dtype) for w in lists]


def _biases(lists, dtype=torch.float32):
    if lists is None:
        return None
    return [None if b is None else torch.tensor(b, dtype=dtype) for b in lists]


def check_example(solve):
    feats = _features([torch.tensor(f) for f in EXAMPLE_FEATURES])
    out = solve(feats, _weights(EXAMPLE_WEIGHTS), _biases(EXAMPLE_BIASES))
    assert set(out.keys()) == {"features", "sizes", "strides", "lateral_params",
                               "num_lateral_params", "num_upsample",
                               "anchors_per_level", "num_anchors"}, sorted(out.keys())
    assert len(out["features"]) == len(EXAMPLE_FUSED)
    for got, exp in zip(out["features"], EXAMPLE_FUSED):
        assert got.dtype == torch.float32
        assert torch.allclose(got, torch.tensor(exp), atol=1e-6), (got.tolist(), exp)
    assert torch.equal(out["sizes"], torch.tensor(EXAMPLE_SIZES, dtype=torch.int64))
    assert torch.equal(out["strides"], torch.tensor(EXAMPLE_STRIDES, dtype=torch.int64))
    assert torch.equal(out["lateral_params"],
                       torch.tensor(EXAMPLE_LATERAL_PARAMS, dtype=torch.int64))
    assert out["num_lateral_params"] == EXAMPLE_NUM_LATERAL_PARAMS
    assert out["num_upsample"] == EXAMPLE_NUM_UPSAMPLE
    assert torch.equal(out["anchors_per_level"],
                       torch.tensor(EXAMPLE_ANCHORS_PER_LEVEL, dtype=torch.int64))
    assert out["num_anchors"] == EXAMPLE_NUM_ANCHORS


check_example.example = True


def _oracle(features, weights, biases, include_p6, anchors_per_loc, strides):
    """逐像素 Python 循环：1x1 卷积、整数最近邻放大、kernel2/stride2 池化。"""
    fs = [f.tolist() for f in features]
    ws = [w.tolist() for w in weights]
    if biases is None:
        biases = [None] * len(fs)
    bs = [None if b is None else b.tolist() for b in biases]
    fpn = len(ws[0])
    lat, params = [], []
    for f, w, b in zip(fs, ws, bs):
        n, c, h, wd = len(f), len(f[0]), len(f[0][0]), len(f[0][0][0])
        out = [[[[0.0] * wd for _ in range(h)] for _ in range(fpn)] for _ in range(n)]
        for ni in range(n):
            for o in range(fpn):
                for y in range(h):
                    for x in range(wd):
                        acc = 0.0
                        for ci in range(c):
                            acc += w[o][ci] * f[ni][ci][y][x]
                        if b is not None:
                            acc += b[o]
                        out[ni][o][y][x] = acc
        lat.append(out)
        params.append(fpn * c + (fpn if b is not None else 0))

    fused = [None] * len(lat)
    fused[-1] = lat[-1]
    for i in range(len(lat) - 2, -1, -1):
        up = fused[i + 1]
        h_t, w_t = len(lat[i][0][0]), len(lat[i][0][0][0])
        h_u, w_u = len(up[0][0]), len(up[0][0][0])
        res = [[[[0.0] * w_t for _ in range(h_t)] for _ in range(fpn)]
               for _ in range(len(lat[i]))]
        for ni in range(len(lat[i])):
            for o in range(fpn):
                for y in range(h_t):
                    sy = y * h_u // h_t
                    for x in range(w_t):
                        sx = x * w_u // w_t
                        res[ni][o][y][x] = lat[i][ni][o][y][x] + up[ni][o][sy][sx]
        fused[i] = res

    num_upsample = len(lat) - 1
    if include_p6:
        p5 = fused[-1]
        h, w = len(p5[0][0]), len(p5[0][0][0])
        oh, ow = h // 2, w // 2
        p6 = [[[[max(p5[ni][o][2 * y][2 * x], p5[ni][o][2 * y][2 * x + 1],
                    p5[ni][o][2 * y + 1][2 * x], p5[ni][o][2 * y + 1][2 * x + 1])
                 for x in range(ow)] for y in range(oh)] for o in range(fpn)]
              for ni in range(len(p5))]
        fused.append(p6)

    base = [2 ** (2 + i) for i in range(len(lat))] if strides is None \
        else [int(s) for s in strides]
    full = base + ([2 * base[-1]] if include_p6 else [])
    sizes = [[len(t[0][0]), len(t[0][0][0])] for t in fused]
    anchors = [anchors_per_loc * h * w for h, w in sizes]
    return {"features": fused, "sizes": sizes, "strides": full,
            "lateral_params": params, "num_lateral_params": sum(params),
            "num_upsample": num_upsample, "anchors_per_level": anchors,
            "num_anchors": sum(anchors)}


def _compare(out, exp, tol=1e-5, tag=""):
    assert len(out["features"]) == len(exp["features"]), (tag, len(out["features"]))
    for t, e in zip(out["features"], exp["features"]):
        e_t = torch.tensor(e, dtype=t.dtype)
        assert t.shape == e_t.shape, (tag, t.shape, e_t.shape)
        assert torch.allclose(t, e_t, atol=tol, rtol=0.0), \
            (tag, (t - e_t).abs().max().item())
    for key in ("sizes", "strides", "lateral_params", "anchors_per_level"):
        assert torch.equal(out[key], torch.tensor(exp[key], dtype=torch.int64)), \
            (tag, key, out[key].tolist(), exp[key])
    for key in ("num_lateral_params", "num_upsample", "num_anchors"):
        assert out[key] == exp[key], (tag, key, out[key], exp[key])


def check_oracle(solve):
    """多级、多 batch、非方形、非整数倍放大、含/不含 P6 的配置全量对拍。"""
    torch.manual_seed(11)
    configs = [
        # 3 级、N=2、6->3 与 3->2 的放大倍率不整；只有中间级无 bias
        ([(2, 2, 8, 6), (2, 3, 4, 3), (2, 1, 2, 2)], 4, [0, 2], False, 3, None),
        # 2 级、非方形、含 P6、自定义 strides
        ([(1, 1, 6, 5), (1, 2, 3, 2)], 2, [], True, 1, [3, 7]),
        # 单级
        ([(1, 3, 4, 3)], 1, [], False, 5, None),
        # 4 级、默认 strides、P6
        ([(1, 2, 8, 8), (1, 2, 4, 4), (1, 2, 2, 2), (1, 2, 2, 2)], 2, [1, 3], True,
         3, None),
    ]
    for ci, (shapes, fpn, bias_idx, include_p6, apl, strides) in enumerate(configs):
        feats = [torch.randn(*s) for s in shapes]
        weights = [torch.randn(fpn, s[1]) for s in shapes]
        biases = [torch.randn(fpn) if i in bias_idx else None
                  for i in range(len(shapes))] if bias_idx else None
        out = solve(feats, weights, biases, include_p6=include_p6,
                    anchors_per_loc=apl, strides=strides)
        exp = _oracle(feats, weights, biases, include_p6, apl, strides)
        _compare(out, exp, tag=f"cfg{ci}")


def check_upsample_rule(solve):
    """最近邻放大用整数规则 src = dst * S_up // S_t（不插值、不夹取）。"""
    # 3x3 -> 5x5：行/列映射都是 [0,0,1,1,2]
    src = torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]])
    dst_shape = torch.zeros(1, 1, 5, 5)
    out = solve([dst_shape, src], [torch.zeros(1, 1), torch.ones(1, 1)])
    fused = out["features"][0][0, 0]
    exp = torch.tensor([[1.0, 1.0, 2.0, 2.0, 3.0],
                        [1.0, 1.0, 2.0, 2.0, 3.0],
                        [4.0, 4.0, 5.0, 5.0, 6.0],
                        [4.0, 4.0, 5.0, 5.0, 6.0],
                        [7.0, 7.0, 8.0, 8.0, 9.0]])
    assert torch.allclose(fused, exp, atol=1e-6), fused.tolist()
    # 2x2 -> 4x4 恰好两倍：每个像素复制成 2x2
    src2 = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    out2 = solve([torch.zeros(1, 1, 4, 4), src2], [torch.zeros(1, 1), torch.ones(1, 1)])
    fused2 = out2["features"][0][0, 0]
    assert torch.allclose(fused2, torch.tensor([[1.0, 1.0, 2.0, 2.0],
                                                [1.0, 1.0, 2.0, 2.0],
                                                [3.0, 3.0, 4.0, 4.0],
                                                [3.0, 3.0, 4.0, 4.0]]), atol=1e-6), \
        fused2.tolist()


def check_p6(solve):
    """P6 = P5 上 kernel2/stride2 最大池化，奇数边向下取整丢末行末列。"""
    f = torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]])
    out = solve([f], [torch.ones(1, 1)], include_p6=True, anchors_per_loc=1)
    assert len(out["features"]) == 2
    p6 = out["features"][1]
    assert p6.shape == (1, 1, 1, 1), p6.shape
    assert abs(p6.item() - 5.0) < 1e-6, p6.item()  # max(1,2,4,5)，不含 7/8/9
    assert out["sizes"].tolist() == [[3, 3], [1, 1]]
    assert out["strides"].tolist() == [4, 8]
    # 4x4 -> 2x2：四个窗口
    g = torch.arange(1.0, 17.0).reshape(1, 1, 4, 4)
    out2 = solve([g], [torch.ones(1, 1)], include_p6=True)
    assert torch.allclose(out2["features"][1], torch.tensor([[[[6.0, 8.0], [14.0, 16.0]]]])), \
        out2["features"][1].tolist()
    assert out2["sizes"].tolist() == [[4, 4], [2, 2]]
    assert out2["anchors_per_level"].tolist() == [48, 12]


def check_params_and_strides(solve):
    """参数量与步长账本：P6 不计侧接参数，步长按给定值原样记账。"""
    shapes = [(1, 4, 16, 16), (1, 3, 8, 8), (1, 2, 4, 4), (1, 1, 2, 2)]
    fpn = 5
    feats = [torch.zeros(*s) for s in shapes]
    weights = [torch.zeros(fpn, s[1]) for s in shapes]
    biases = [torch.zeros(fpn), None, torch.zeros(fpn), None]
    out = solve(feats, weights, biases, include_p6=True, anchors_per_loc=3)
    assert out["lateral_params"].tolist() == [25, 15, 15, 5], out["lateral_params"].tolist()
    assert out["num_lateral_params"] == 60
    assert out["num_upsample"] == 3
    assert out["strides"].tolist() == [4, 8, 16, 32, 64]
    assert out["sizes"].tolist() == [[16, 16], [8, 8], [4, 4], [2, 2], [1, 1]]
    assert out["anchors_per_level"].tolist() == [768, 192, 48, 12, 3]
    assert out["num_anchors"] == 1023
    # 自定义步长原样记账，不从 H/W 反推；P6 = 2 * 最后一级
    out2 = solve(feats, weights, biases, include_p6=True, anchors_per_loc=1,
                 strides=[3, 7, 13, 29])
    assert out2["strides"].tolist() == [3, 7, 13, 29, 58], out2["strides"].tolist()
    # 不含 P6 时账本长度等于级数
    out3 = solve(feats, weights, biases, include_p6=False)
    assert out3["strides"].numel() == 4 and out3["lateral_params"].numel() == 4
    assert out3["num_upsample"] == 3


def check_dtype_and_immutability(solve):
    feats64 = _features([torch.tensor(f) for f in EXAMPLE_FEATURES], dtype=torch.float64)
    w64 = _weights(EXAMPLE_WEIGHTS, dtype=torch.float64)
    b64 = _biases(EXAMPLE_BIASES, dtype=torch.float64)
    snap = [t.clone() for t in feats64], [t.clone() for t in w64], \
        [None if b is None else b.clone() for b in b64]
    out = solve(feats64, w64, b64)
    for got, exp in zip(out["features"], EXAMPLE_FUSED):
        assert got.dtype == torch.float64, got.dtype
        assert torch.allclose(got, torch.tensor(exp, dtype=torch.float64), atol=1e-12)
    for a, b in zip(feats64, snap[0]):
        assert torch.equal(a, b)
    for a, b in zip(w64, snap[1]):
        assert torch.equal(a, b)
    for a, b in zip(b64, snap[2]):
        if a is not None:
            assert torch.equal(a, b)


def check_errors(solve):
    f = [torch.zeros(1, 2, 4, 4), torch.zeros(1, 1, 2, 2)]
    w = [torch.zeros(2, 2), torch.zeros(2, 1)]

    def bad(fn, desc):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError(f"应抛 ValueError: {desc}")

    bad(lambda: solve(f[0], w), "features 非 list")
    bad(lambda: solve([], []), "features 为空")
    bad(lambda: solve([torch.zeros(1, 2, 4)], w), "feature 非四维")
    bad(lambda: solve([[1.0]], w), "feature 非张量")
    bad(lambda: solve([torch.zeros(1, 2, 0, 4), f[1]], w), "feature H/W < 1")
    bad(lambda: solve(f, w[:1]), "lateral_weight 长度")
    bad(lambda: solve(f, [torch.zeros(2), w[1]]), "lateral_weight 非二维")
    bad(lambda: solve(f, [[[1.0, 0.0]], w[1]]), "lateral_weight 非张量")
    bad(lambda: solve(f, [torch.zeros(2, 3), w[1]]), "lateral_weight 通道不符")
    bad(lambda: solve(f, [torch.zeros(2, 2), torch.zeros(3, 1)]), "fpn_dim 不一致")
    bad(lambda: solve(f, [torch.zeros(0, 2), torch.zeros(0, 1)]), "fpn_dim < 1")
    bad(lambda: solve(f, w, [torch.zeros(2)]), "lateral_bias 长度")
    bad(lambda: solve(f, w, [torch.zeros(3), None]), "lateral_bias 形状")
    bad(lambda: solve(f, w, [[0.0, 0.0], None]), "lateral_bias 非张量")
    bad(lambda: solve(f, w, strides=[4]), "strides 长度")
    bad(lambda: solve(f, w, strides=[0, 8]), "strides 非正")
    bad(lambda: solve(f, w, anchors_per_loc=0), "anchors_per_loc < 1")
    bad(lambda: solve([torch.zeros(1, 1, 1, 1)], [torch.zeros(1, 1)], include_p6=True),
        "P5 小于 2 无法池化")
    bad(lambda: solve([torch.zeros(1, 1, 3, 1)], [torch.zeros(1, 1)], include_p6=True),
        "P5 宽为 1 无法池化")
