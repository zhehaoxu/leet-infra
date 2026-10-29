import math

import torch

# 示例（实测回填）：C=2, H=W=3，三个采样点
EXAMPLE_FEATURE = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                   [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0]]]
EXAMPLE_POINTS = [[[1.0, 1.0], [0.0, 0.0], [1.0, 1.0]]]
EXAMPLE_OFFSETS = [[[0.5, 0.0], [-0.5, 0.0], [0.5, 0.5]]]
EXAMPLE_WEIGHTS = [[0.5, 0.25, 0.25]]
EXAMPLE_SAMPLED = [[[5.5, 55.0], [0.5, 5.0], [7.0, 70.0]]]
EXAMPLE_AGGREGATED = [4.625, 46.25]


def _feature(vals=EXAMPLE_FEATURE, dtype=torch.float64):
    return torch.tensor(vals, dtype=dtype)


def _pts(vals, dtype=torch.float64):
    return torch.tensor(vals, dtype=dtype)


def check_example(solve):
    feat = _feature()
    sampled = solve(feat, _pts(EXAMPLE_POINTS), _pts(EXAMPLE_OFFSETS))
    assert sampled.shape == (1, 3, 2) and sampled.dtype == torch.float64, \
        (sampled.shape, sampled.dtype)
    exp = torch.tensor(EXAMPLE_SAMPLED, dtype=torch.float64)
    assert torch.allclose(sampled, exp, atol=1e-12), sampled.tolist()
    agg = solve(feat, _pts(EXAMPLE_POINTS), _pts(EXAMPLE_OFFSETS), _pts(EXAMPLE_WEIGHTS))
    assert agg.shape == (1, 2) and agg.dtype == torch.float64, (agg.shape, agg.dtype)
    exp_a = torch.tensor([EXAMPLE_AGGREGATED], dtype=torch.float64)
    assert torch.allclose(agg, exp_a, atol=1e-12), agg.tolist()


check_example.example = True


def _bilinear(feat, c, x, y):
    h, w = len(feat[0]), len(feat[0][0])
    x0, y0 = math.floor(x), math.floor(y)
    fx, fy = x - x0, y - y0
    total = 0.0
    # 角点按 (1,1) -> (1,0) -> (0,1) -> (0,0) 的相反顺序累加，浮点舍入路径与参考解不同
    for dy, dx, wgt in ((1, 1, fx * fy), (1, 0, (1.0 - fx) * fy),
                        (0, 1, fx * (1.0 - fy)), (0, 0, (1.0 - fx) * (1.0 - fy))):
        xi, yi = x0 + dx, y0 + dy
        if 0 <= xi < w and 0 <= yi < h:
            total += wgt * feat[c][yi][xi]
    return total


def _oracle(feature, points, offsets, weights):
    feat = feature.tolist()
    n, k = points.shape[0], points.shape[1]
    c = feature.shape[0]
    sampled = [[[_bilinear(feat, cc, float(points[i][j][0] + offsets[i][j][0]),
                          float(points[i][j][1] + offsets[i][j][1]))
                 for cc in range(c)] for j in range(k)] for i in range(n)]
    if weights is None:
        return sampled
    wl = weights.tolist()
    return [[sum(wl[i][j] * sampled[i][j][cc] for j in range(k)) for cc in range(c)]
            for i in range(n)]


def check_oracle(solve):
    """任意越界/小数坐标、含与不含权重，与纯 Python 逐角 oracle 全量对拍。"""
    torch.manual_seed(7)
    for ci in range(4):
        c = 1 + ci % 3
        h, w = 2 + ci, 3 + ci
        feat = torch.randn(c, h, w, dtype=torch.float64)
        n, k = 4, 6
        pts = torch.empty(n, k, 2, dtype=torch.float64)
        pts[..., 0] = torch.rand(n, k, dtype=torch.float64) * (w + 4.0) - 2.0
        pts[..., 1] = torch.rand(n, k, dtype=torch.float64) * (h + 4.0) - 2.0
        off = (torch.rand(n, k, 2, dtype=torch.float64) - 0.5) * 3.0
        exp = torch.tensor(_oracle(feat, pts, off, None), dtype=torch.float64)
        got = solve(feat, pts, off)
        assert got.shape == (n, k, c), (got.shape, (n, k, c))
        assert torch.allclose(got, exp, atol=1e-12), (ci, (got - exp).abs().max().item())
        wgt = torch.rand(n, k, dtype=torch.float64) * 2.0
        exp_a = torch.tensor(_oracle(feat, pts, off, wgt), dtype=torch.float64)
        got_a = solve(feat, pts, off, wgt)
        assert got_a.shape == (n, c), (got_a.shape, (n, c))
        assert torch.allclose(got_a, exp_a, atol=1e-12), (ci, (got_a - exp_a).abs().max().item())
        # float32 输入：结果 dtype 为 float32，与 oracle 在 1e-5 内一致
        f32 = feat.to(torch.float32)
        got32 = solve(f32, pts.to(torch.float32), off.to(torch.float32))
        assert got32.dtype == torch.float32
        assert torch.allclose(got32, exp.to(torch.float32), atol=1e-5), \
            (ci, (got32 - exp.to(torch.float32)).abs().max().item())


def check_integer_identity(solve):
    """整数坐标 + 零偏移必须原样取出像素，且 (x=列, y=行) 不能对调。"""
    feat = _feature()
    coords = [[0, 0], [2, 0], [0, 2], [2, 2], [2, 1]]
    pts = _pts([[[float(x), float(y)] for x, y in coords]])
    zero = torch.zeros_like(pts)
    got = solve(feat, pts, zero)
    exp = torch.tensor([[[feat[0, y, x].item(), feat[1, y, x].item()] for x, y in coords]],
                       dtype=torch.float64)
    assert torch.equal(got, exp), (got.tolist(), exp.tolist())
    # 显式钉住一例：(x=2, y=1) 是第 1 行第 2 列 = 6 / 60
    assert got[0, 4].tolist() == [6.0, 60.0], got[0, 4].tolist()


def check_zero_padding(solve):
    """界外角按 0 参与，不夹取、不复制边缘。"""
    feat = _feature()
    f00 = feat[:, 0, 0]
    f01 = feat[:, 0, 1]
    f10 = feat[:, 1, 0]
    f22 = feat[:, 2, 2]

    def run(x, y):
        return solve(feat, _pts([[[x, y]]]), torch.zeros(1, 1, 2, dtype=torch.float64))[0, 0]

    # x=-0.5, y=0 -> 只有 (0,0) 这个角在界内，权重 0.5
    assert torch.allclose(run(-0.5, 0.0), 0.5 * f00, atol=1e-12), run(-0.5, 0.0).tolist()
    # x=-1.0 -> 权重全落在界外的 (x0=-1) 上，(0,0) 的权重为 0 -> 0
    assert torch.equal(run(-1.0, 0.0), torch.zeros(2, dtype=torch.float64)), run(-1.0, 0.0).tolist()
    # 左上角外一格：x0=-1 的两个角越界，y=0/1 两个角存活，各 0.25
    assert torch.allclose(run(-0.5, 0.5), 0.25 * (f00 + feat[:, 1, 0]),
                          atol=1e-12), run(-0.5, 0.5).tolist()
    # 右边界外半格：W-0.5 -> 只有 (W-1,0) 权重 0.5
    assert torch.allclose(run(2.5, 0.0), 0.5 * feat[:, 0, 2],
                          atol=1e-12), run(2.5, 0.0).tolist()
    # y 方向同理
    assert torch.allclose(run(0.0, 2.5), 0.5 * feat[:, 2, 0], atol=1e-12), run(0.0, 2.5).tolist()
    # 完全在内部：0.5*(f00 + f01)
    assert torch.allclose(run(0.5, 0.0), 0.5 * (f00 + f01), atol=1e-12), run(0.5, 0.0).tolist()
    # 内部半格：四角等权
    assert torch.allclose(run(0.5, 0.5), 0.25 * (f00 + f01 + f10 + feat[:, 1, 1]),
                          atol=1e-12), run(0.5, 0.5).tolist()
    # 右下角外半格：只剩 (2,2) 权重 0.25
    assert torch.allclose(run(2.5, 2.5), 0.25 * f22, atol=1e-12), run(2.5, 2.5).tolist()
    # 远处：全 0，不抛错
    assert torch.equal(run(50.0, -20.0), torch.zeros(2, dtype=torch.float64)), run(50.0, -20.0).tolist()


def check_offsets_applied(solve):
    """采样位置是 point + offset，不是 point 本身。"""
    feat = _feature()
    # (0,0) + (2,1) -> 恰好取 (x=2, y=1) = 6 / 60
    got = solve(feat, _pts([[[0.0, 0.0]]]), _pts([[[2.0, 1.0]]]))
    assert torch.equal(got, torch.tensor([[[6.0, 60.0]]], dtype=torch.float64)), got.tolist()
    # 如果忽略 offset，会得到 1 / 10
    assert got[0, 0].tolist() != [1.0, 10.0]
    # (0,0) + (-0.5,-0.5) -> 只有 (0,0) 存活，权重 0.25
    got = solve(feat, _pts([[[0.0, 0.0]]]), _pts([[[-0.5, -0.5]]]))
    assert torch.allclose(got, torch.tensor([[[0.25, 2.5]]], dtype=torch.float64),
                          atol=1e-12), got.tolist()
    # 负向半格但行内：y=0.5 时四个角都参与
    got = solve(feat, _pts([[[0.0, 0.0]]]), _pts([[[0.0, 0.5]]]))
    assert torch.allclose(got, torch.tensor([[[2.5, 25.0]]], dtype=torch.float64),
                          atol=1e-12), got.tolist()


def check_weights(solve):
    """聚合是加权和，不做归一化；K=1 时权重就是普通缩放。"""
    feat = _feature()
    pts = _pts(EXAMPLE_POINTS)
    off = _pts(EXAMPLE_OFFSETS)
    sampled = solve(feat, pts, off)
    # 权重和为 2：不归一化 -> 6.0 / 60.0；归一化会得到 3.0 / 30.0
    got = solve(feat, pts, off, _pts([[1.0, 1.0, 0.0]]))
    assert torch.allclose(got, torch.tensor([[6.0, 60.0]], dtype=torch.float64),
                          atol=1e-12), got.tolist()
    assert torch.allclose(got, sampled[0, 0] + sampled[0, 1], atol=1e-12)
    # 全零权重 -> 0
    got0 = solve(feat, pts, off, torch.zeros(1, 3, dtype=torch.float64))
    assert torch.equal(got0, torch.zeros(1, 2, dtype=torch.float64)), got0.tolist()
    # K=1，权重 2.0 -> 采样值翻倍
    one = solve(feat, pts[:, :1], off[:, :1], _pts([[2.0]]))
    assert torch.allclose(one, 2.0 * sampled[:, :1].squeeze(1), atol=1e-12), one.tolist()
    # 负权重也照公式加起来，不裁剪
    neg = solve(feat, pts[:, :2], off[:, :2], _pts([[1.0, -1.0]]))
    assert torch.allclose(neg, (sampled[:, 0] - sampled[:, 1]), atol=1e-12), neg.tolist()


def check_dtype_and_immutability(solve):
    feat = _feature()
    pts = _pts(EXAMPLE_POINTS)
    off = _pts(EXAMPLE_OFFSETS)
    wgt = _pts(EXAMPLE_WEIGHTS)
    snap = [t.clone() for t in (feat, pts, off, wgt)]
    out64 = solve(feat, pts, off, wgt)
    out32 = solve(feat.to(torch.float32), pts.to(torch.float32), off.to(torch.float32),
                  wgt.to(torch.float32))
    assert out32.dtype == torch.float32 and out64.dtype == torch.float64
    assert torch.allclose(out32.to(torch.float64), out64, atol=1e-5), \
        (out32.tolist(), out64.tolist())
    for a, b in zip((feat, pts, off, wgt), snap):
        assert torch.equal(a, b), "输入被原地修改"
    # 聚合输出 dtype 跟随 feature
    assert solve(feat.to(torch.float32), pts.to(torch.float32),
                 off.to(torch.float32)).dtype == torch.float32


def check_empty(solve):
    feat = _feature()
    pts = torch.zeros(0, 2, 2, dtype=torch.float64)
    off = torch.zeros(0, 2, 2, dtype=torch.float64)
    got = solve(feat, pts, off)
    assert got.shape == (0, 2, 2) and got.dtype == torch.float64, (got.shape, got.dtype)
    agg = solve(feat, pts, off, torch.zeros(0, 2, dtype=torch.float64))
    assert agg.shape == (0, 2) and agg.dtype == torch.float64, (agg.shape, agg.dtype)


def check_errors(solve):
    feat = _feature()
    pts = _pts([[[0.0, 0.0]]])
    off = torch.zeros(1, 1, 2, dtype=torch.float64)
    wgt = _pts([[1.0]])

    def bad(fn, desc):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError(f"应抛 ValueError: {desc}")

    bad(lambda: solve(feat.tolist(), pts, off), "feature 非张量")
    bad(lambda: solve(feat[0], pts, off), "feature 二维")
    bad(lambda: solve(torch.zeros(0, 2, 2), pts, off), "C 为 0")
    bad(lambda: solve(torch.zeros(2, 0, 2), pts, off), "H 为 0")
    bad(lambda: solve(feat.to(torch.int64), pts.to(torch.int64), off.to(torch.int64)), "整型 dtype")
    bad(lambda: solve(feat, pts.tolist(), off), "points 非张量")
    bad(lambda: solve(feat, _pts([[0.0, 0.0]]), off), "points 二维")
    bad(lambda: solve(feat, _pts([[[0.0, 0.0, 0.0]]]), off), "points 最后一维不是 2")
    bad(lambda: solve(feat, pts, torch.zeros(1, 2, 2, dtype=torch.float64)), "points/offsets 形状不符")
    bad(lambda: solve(feat, pts.to(torch.float32), off.to(torch.float32)), "points dtype 与 feature 不符")
    bad(lambda: solve(feat, pts, off.to(torch.float32)), "offsets dtype 与 feature 不符")
    bad(lambda: solve(feat, torch.zeros(1, 0, 2, dtype=torch.float64),
                      torch.zeros(1, 0, 2, dtype=torch.float64)), "K 为 0")
    bad(lambda: solve(feat, pts, off, [[1.0]]), "weights 非张量")
    bad(lambda: solve(feat, pts, off, _pts([[1.0, 2.0]])), "weights 形状不符")
    bad(lambda: solve(feat, pts, off, wgt.to(torch.float32)), "weights dtype 不符")
    bad(lambda: solve(feat, _pts([[[float("nan"), 0.0]]]), off), "points 含 NaN")
    bad(lambda: solve(feat, pts, _pts([[[float("inf"), 0.0]]])), "offsets 含 Inf")
    bad(lambda: solve(feat, pts, off, _pts([[float("nan")]])), "weights 含 NaN")
