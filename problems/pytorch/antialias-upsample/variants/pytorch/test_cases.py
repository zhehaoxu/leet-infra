import math

import torch
import torch.nn.functional as F

# 示例期望值：x=[[1..12]] (1,1,3,4)，size=(2,2)，bilinear 非 AA。
# half-pixel：sy=3/2、sx=4/2，输出权重全为二进制精确值，手算与原生实测一致。
EXAMPLE_OUT = [[2.5, 4.5], [8.5, 10.5]]

EX_X = [[1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0]]


def _det(*shape, dtype=torch.float32):
    """确定性输入（不调用任何随机数发生器）：arange 的 sin/cos 组合花纹，值域约 [-4.5, 4.5]。"""
    n = 1
    for s in shape:
        n *= s
    idx = torch.arange(n, dtype=torch.float64)
    v = torch.sin(idx * 0.37) * 3.0 + torch.cos(idx * 0.11) * 1.5
    return v.reshape(shape).to(dtype)


def _native(x, size, mode, antialias=False, align_corners=False):
    return F.interpolate(x, size=size, mode=mode, antialias=antialias,
                         align_corners=align_corners)


def _oracle_bilinear_2d(x2d, oh, ow, antialias, align_corners=False):
    """逐输出像素的纯 Python 双精度循环 oracle（bilinear）。

    与解法的"权重矩阵 + matmul"走完全不同的路径：
    - antialias=True 采用原生 AA 口径：center = scale*(dst+0.5)（half-pixel，
      即使 align_corners=True 也如此），support = max(1, scale)，
      权重 = tri((tap-center+0.5)*invscale) 后按和归一化；
    - antialias=False：align_corners=True 时 src=scale*dst（不 clamp），
      False 时 src=scale*(dst+0.5)-0.5 且线性核下 clamp 到 >=0，2 tap 线性插值。
    """
    h_in, w_in = len(x2d), len(x2d[0])

    def taps_weights(n_in, n_out, aa):
        # scale 口径与原生一致：align_corners=True 用 (n_in-1)/(n_out-1)（含 AA 路径，
        # 即 AA 的 half-pixel center 也乘这个 ca scale）；n_out=1 时 scale=0
        if align_corners:
            scale = (n_in - 1) / (n_out - 1) if n_out > 1 else 0.0
        else:
            scale = n_in / n_out
        rows = []
        for i in range(n_out):
            if aa:
                support = scale if scale >= 1.0 else 1.0  # bilinear: interp_size/2 = 1
                invscale = 1.0 / scale if scale >= 1.0 else 1.0
                center = scale * (i + 0.5)
                lo = max(int(math.floor(center - support + 0.5)), 0)
                hi = min(int(math.floor(center + support + 0.5)), n_in)
                tw = []
                for t in range(lo, hi):
                    w = 1.0 - abs((t - center + 0.5) * invscale)
                    if w > 0.0:
                        tw.append((t, w))
                tot = sum(w for _, w in tw)
                rows.append([(t, w / tot) for t, w in tw] if tot != 0.0 else tw)
            else:
                if align_corners:
                    src = scale * i
                else:
                    src = max(scale * (i + 0.5) - 0.5, 0.0)
                k = min(int(math.floor(src)), n_in - 1)
                lam = min(max(src - k, 0.0), 1.0)
                k1 = k + 1 if k < n_in - 1 else k
                rows.append([(k, 1.0 - lam), (k1, lam)])
        return rows

    rows_y = taps_weights(h_in, oh, antialias)
    rows_x = taps_weights(w_in, ow, antialias)
    res = [[0.0] * ow for _ in range(oh)]
    for i in range(oh):
        for j in range(ow):
            acc = 0.0
            for ty, wy in rows_y[i]:
                for tx, wx in rows_x[j]:
                    acc += wy * wx * float(x2d[ty][tx])
            res[i][j] = acc
    return res


def check_example(solve):
    x = torch.tensor(EX_X, dtype=torch.float32).reshape(1, 1, 3, 4)
    out = solve(x, (2, 2), mode="bilinear")  # 默认 antialias=False, align_corners=False
    assert out.shape == (1, 1, 2, 2), out.shape
    assert out.dtype == torch.float32, out.dtype
    exp = torch.tensor(EXAMPLE_OUT, dtype=torch.float32).reshape(1, 1, 2, 2)
    assert torch.allclose(out, exp, atol=1e-6, rtol=1e-6), (out.tolist(), exp.tolist())
    # 同一输入交给原生交叉验证（示例本身也应对齐原生）
    assert torch.allclose(out, _native(x, (2, 2), "bilinear"), atol=1e-5, rtol=1e-5)


check_example.example = True


def check_native_battery(solve):
    """与 F.interpolate CPU 原生逐点对齐：bilinear/bicubic/trilinear x AA 两态 x
    align_corners 两态 x 放大/缩小/混合/恒等，float32 用 1e-5、float64 用 1e-12。"""
    cfg32 = [
        # (输入形状, size, mode, antialias, align_corners)
        ((1, 1, 3, 4), (2, 2), "bilinear", False, False),
        ((1, 1, 3, 4), (2, 2), "bilinear", True, False),
        ((2, 3, 12, 13), (5, 7), "bilinear", False, False),
        ((2, 3, 12, 13), (5, 7), "bilinear", True, False),
        ((2, 3, 12, 13), (5, 7), "bilinear", True, True),
        ((2, 3, 12, 13), (5, 7), "bilinear", False, True),
        ((2, 3, 12, 13), (5, 7), "bicubic", False, False),
        ((2, 3, 12, 13), (5, 7), "bicubic", True, False),
        ((2, 3, 12, 13), (5, 7), "bicubic", True, True),
        ((2, 3, 12, 13), (5, 7), "bicubic", False, True),
        ((1, 2, 4, 9), (11, 4), "bilinear", True, False),   # H 放大 / W 缩小混合
        ((1, 2, 4, 9), (11, 4), "bicubic", True, False),
        ((2, 2, 6, 6), (6, 6), "bicubic", True, False),     # 恒等
        ((2, 3, 7, 5), (9, 3), "bilinear", True, False),    # 单侧 out=1 以外的小尺寸
        ((1, 1, 2, 3, 4), (3, 4, 5), "trilinear", False, False),
        ((1, 1, 5, 6, 7), (2, 2, 2), "trilinear", False, False),
        ((1, 1, 5, 6, 7), (2, 2, 2), "trilinear", False, True),
        ((2, 2, 4, 5, 6), (7, 3, 9), "trilinear", False, False),
    ]
    for shape, size, mode, aa, ac in cfg32:
        x = _det(*shape)
        got = solve(x, size, mode=mode, antialias=aa, align_corners=ac)
        exp = _native(x, size, mode, antialias=aa, align_corners=ac)
        assert got.shape == exp.shape and got.dtype == exp.dtype, (shape, size, mode, aa, ac)
        assert torch.allclose(got, exp, atol=1e-5, rtol=1e-5), \
            (shape, size, mode, aa, ac, (got - exp).abs().max().item())
    cfg64 = [
        ((2, 3, 12, 13), (5, 7), "bilinear", True, False),
        ((2, 3, 12, 13), (5, 7), "bicubic", True, True),
        ((2, 3, 12, 13), (5, 7), "bicubic", False, False),
        ((1, 1, 5, 6, 7), (2, 2, 2), "trilinear", False, False),
        ((1, 2, 4, 9), (11, 4), "bicubic", True, False),
    ]
    for shape, size, mode, aa, ac in cfg64:
        x = _det(*shape, dtype=torch.float64)
        got = solve(x, size, mode=mode, antialias=aa, align_corners=ac)
        exp = _native(x, size, mode, antialias=aa, align_corners=ac)
        assert torch.allclose(got, exp, atol=1e-12, rtol=1e-12), \
            (shape, size, mode, aa, ac, (got - exp).abs().max().item())


def check_pixel_loop_oracle_aa(solve):
    """独立逐像素循环 oracle：bilinear 的 AA 两态（含 align_corners=True 的
    half-pixel center 特殊口径），oracle 全程 float64 纯 Python 循环。"""
    cases = [
        # (H, W, oh, ow, antialias, align_corners)
        (8, 9, 3, 4, False, False),
        (8, 9, 3, 4, True, False),
        (7, 5, 4, 6, True, False),   # H 缩小 / W 放大混合
        (4, 4, 7, 7, False, False),  # 放大
        (6, 6, 2, 2, True, True),    # align_corners=True + AA（center 用 half-pixel）
        (6, 6, 2, 2, False, True),
    ]
    for h, w, oh, ow, aa, ac in cases:
        x = _det(1, 1, h, w)
        got = solve(x, (oh, ow), mode="bilinear", antialias=aa, align_corners=ac)
        exp = _oracle_bilinear_2d(x[0, 0].tolist(), oh, ow, aa, ac)
        exp_t = torch.tensor(exp, dtype=torch.float32).reshape(1, 1, oh, ow)
        assert torch.allclose(got, exp_t, atol=1e-5, rtol=1e-5), \
            (h, w, oh, ow, aa, ac, (got - exp_t).abs().max().item())


def check_identity_bitwise(solve):
    """恒等尺寸（n_out == n_in）应逐位透传，与 AA 两态、mode 无关。"""
    for mode in ("bilinear", "bicubic"):
        for aa in (False, True):
            x = _det(1, 2, 3, 4)
            out = solve(x, (3, 4), mode=mode, antialias=aa)
            assert torch.equal(out, x), (mode, aa)
    x3 = _det(1, 1, 2, 3, 4)
    out3 = solve(x3, (2, 3, 4), mode="trilinear")
    assert torch.equal(out3, x3)


def check_upscale_aa_equivalence(solve):
    """bilinear + half-pixel 放大：AA 与非 AA 数学上等价（同 tap 同权重），
    参考实现逐位一致；bicubic 放大时 a=-0.5 vs a=-0.75 导致两态不同。"""
    x = _det(1, 2, 3, 4)
    for size in ((6, 8), (7, 9)):
        a = solve(x, size, mode="bilinear", antialias=False)
        b = solve(x, size, mode="bilinear", antialias=True)
        assert torch.equal(a, b), size
        # 与原生各自的 AA 状态也对齐
        assert torch.allclose(b, _native(x, size, "bilinear", antialias=True), atol=1e-5, rtol=1e-5)
        assert torch.allclose(a, _native(x, size, "bilinear", antialias=False), atol=1e-5, rtol=1e-5)
    # bicubic 放大两态必须不同（Keys a=-0.5 vs -0.75）
    xb = _det(1, 1, 4, 4)
    cb = solve(xb, (8, 8), mode="bicubic", antialias=False)
    ca = solve(xb, (8, 8), mode="bicubic", antialias=True)
    assert not torch.allclose(cb, ca, atol=1e-4), \
        "bicubic 放大时 antialias 两态应不同"
    # 各自仍对齐原生
    assert torch.allclose(cb, _native(xb, (8, 8), "bicubic", antialias=False), atol=1e-5, rtol=1e-5)
    assert torch.allclose(ca, _native(xb, (8, 8), "bicubic", antialias=True), atol=1e-5, rtol=1e-5)


def check_align_corners_one_probe(solve):
    """align_corners=True 且某维 out=1 时 scale=0，整维映射到源索引 0。"""
    x = _det(1, 2, 3, 4)
    out = solve(x, (1, 4), mode="bilinear", align_corners=True)
    assert torch.equal(out, x[..., 0:1, :])
    out2 = solve(x, (3, 1), mode="bilinear", align_corners=True)
    assert torch.equal(out2, x[..., 0:1])
    out3 = solve(x, (1, 1), mode="bilinear", align_corners=True)
    assert torch.equal(out3, x[..., 0:1, 0:1])
    # 原生口径一致
    assert torch.equal(_native(x, (1, 4), "bilinear", align_corners=True), x[..., 0:1, :])


def check_channels_batch_independence(solve):
    """batch / channel 维只是 spectators：整体结果 == 逐 (b, c) 切片结果。"""
    x = _det(2, 3, 7, 8)
    got = solve(x, (4, 5), mode="bicubic", antialias=True)
    for b in range(2):
        for c in range(3):
            solo = solve(x[b, c].reshape(1, 1, 7, 8), (4, 5), mode="bicubic", antialias=True)
            assert torch.allclose(got[b, c].reshape(1, 1, 4, 5), solo, atol=1e-6, rtol=1e-6), \
                (b, c)
    # batch 维同理（用 trilinear 覆盖 5-D 路径）
    x5 = _det(2, 2, 3, 4, 5)
    got5 = solve(x5, (2, 3, 4), mode="trilinear")
    for b in range(2):
        solo5 = solve(x5[b].reshape(1, 2, 3, 4, 5), (2, 3, 4), mode="trilinear")
        assert torch.allclose(got5[b], solo5[0], atol=1e-6, rtol=1e-6), b


def check_error_cases(solve):
    x4 = _det(1, 1, 3, 4)
    x5 = _det(1, 1, 2, 3, 4)

    def expect_value_error(fn, desc):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError(f"应抛 ValueError: {desc}")

    expect_value_error(lambda: solve(x4, (2, 2), mode="nearest"), "非法 mode")
    expect_value_error(lambda: solve(x5, (2, 2, 2), mode="trilinear", antialias=True),
                       "antialias+trilinear")
    expect_value_error(lambda: solve(_det(1, 2, 3), (2, 2), mode="bilinear"),
                       "bilinear 需要 4-D")
    expect_value_error(lambda: solve(x4, (2, 2), mode="trilinear"), "trilinear 需要 5-D")
    expect_value_error(lambda: solve(x4, (2,), mode="bilinear"), "size 长度不符")
    expect_value_error(lambda: solve(x4, (2, 0), mode="bilinear"), "size 含非正整数")
    expect_value_error(lambda: solve(x4.to(torch.int64), (2, 2), mode="bilinear"),
                       "dtype 必须是 float32/float64")
