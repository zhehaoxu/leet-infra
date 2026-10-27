import torch
import torch.nn.functional as F

# 示例：3×3 手写图 (1,1,3,3)，kernel=2、stride=2（唯一窗口 [1,2,4,5]，p=2）
EX_X = [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]]
EX_K, EX_S, EX_P = 2, 2, 2
# 期望值（实测回填）：唯一窗口 [1,2,4,5] → lppool=√46、max=5、indices=原图 (1,1) 展平偏移 4
EXAMPLE_LPPOOL = [[[[6.7823300362]]]]   # (1,1,1,1)，float32 实测值
EXAMPLE_OUT = [[[[5.0]]]]               # (1,1,1,1)
EXAMPLE_INDICES = [[[[4]]]]             # (1,1,1,1)，int64


def _pair(v):
    if isinstance(v, int) and not isinstance(v, bool):
        return (v, v)
    if isinstance(v, (tuple, list)) and len(v) in (1, 2):
        return (v[0], v[0]) if len(v) == 1 else (v[0], v[1])
    raise ValueError(v)


def _det(*shape, seed=0, lo=-1.0, hi=1.0):
    """无 RNG 的确定性伪随机张量：黄金比例哈希，CPU/float32 下完全可复现。"""
    numel = 1
    for d in shape:
        numel *= d
    base = torch.arange(numel, dtype=torch.float64)
    v = ((base + 1.0) * (seed + 1) * 0.6180339887498949) % 1.0
    return (v * (hi - lo) + lo).to(torch.float32).reshape(*shape)


def _oracle_lppool2d(x, k, s=None, p=2.0, pad=0):
    """独立多重循环 oracle：零填充后逐窗累加 |x|^p 再开 p 次方（纯 python 浮点）。"""
    kh, kw = _pair(k)
    sh, sw = (kh, kw) if s is None else _pair(s)
    ph, pw = (0, 0) if pad is None else _pair(pad)
    n, c, h, w = x.shape
    xl = x.tolist()
    oh = (h + 2 * ph - kh) // sh + 1
    ow = (w + 2 * pw - kw) // sw + 1
    inv = 1.0 / float(p)
    out = [[[[0.0] * ow for _ in range(oh)] for _ in range(c)] for _ in range(n)]
    for ni in range(n):
        for ci in range(c):
            for i in range(oh):
                for j in range(ow):
                    acc = 0.0
                    for a in range(kh):
                        for b in range(kw):
                            rr, rc = i * sh + a - ph, j * sw + b - pw
                            if 0 <= rr < h and 0 <= rc < w:
                                acc += abs(xl[ni][ci][rr][rc]) ** float(p)
                    out[ni][ci][i][j] = acc ** inv
    return torch.tensor(out, dtype=x.dtype)


def _oracle_lppool1d(x, k, s=None, p=2.0, pad=0):
    kh, = (_pair(k)[0],)
    sh = _pair(s)[0] if s is not None else kh
    ph = _pair(pad)[0] if pad is not None else 0
    n, c, l_in = x.shape
    xl = x.tolist()
    ol = (l_in + 2 * ph - kh) // sh + 1
    inv = 1.0 / float(p)
    out = [[[0.0] * ol for _ in range(c)] for _ in range(n)]
    for ni in range(n):
        for ci in range(c):
            for i in range(ol):
                acc = 0.0
                for a in range(kh):
                    rr = i * sh + a - ph
                    if 0 <= rr < l_in:
                        acc += abs(xl[ni][ci][rr]) ** float(p)
                out[ni][ci][i] = acc ** inv
    return torch.tensor(out, dtype=x.dtype)


def _oracle_maxpool(x, k, s=None, pad=0):
    """独立多重循环 oracle：−inf 填充，逐窗取 max 与首个 argmax 的原图展平偏移。"""
    kh, kw = _pair(k)
    sh, sw = (kh, kw) if s is None else _pair(s)
    ph, pw = (0, 0) if pad is None else _pair(pad)
    n, c, h, w = x.shape
    xl = x.tolist()
    oh = (h + 2 * ph - kh) // sh + 1
    ow = (w + 2 * pw - kw) // sw + 1
    out = [[[[None] * ow for _ in range(oh)] for _ in range(c)] for _ in range(n)]
    idx = [[[[None] * ow for _ in range(oh)] for _ in range(c)] for _ in range(n)]
    for ni in range(n):
        for ci in range(c):
            for i in range(oh):
                for j in range(ow):
                    best = None
                    for a in range(kh):
                        for b in range(kw):
                            rr, rc = i * sh + a - ph, j * sw + b - pw
                            v = xl[ni][ci][rr][rc] if (0 <= rr < h and 0 <= rc < w) else float("-inf")
                            if best is None or v > best:
                                best, bestr, bestc = v, rr, rc
                    out[ni][ci][i][j] = best
                    idx[ni][ci][i][j] = bestr * w + bestc
    return torch.tensor(out, dtype=x.dtype), torch.tensor(idx, dtype=torch.long)


def _oracle_unpool(x, indices, output_size):
    """独立多重循环 oracle：逐元素写回 indices 指向的源位置，其余 0。"""
    n, c, hs, ws = x.shape
    oh, ow = output_size
    xl, il = x.tolist(), indices.tolist()
    out = [[[[0.0] * ow for _ in range(oh)] for _ in range(c)] for _ in range(n)]
    for ni in range(n):
        for ci in range(c):
            for i in range(hs):
                for j in range(ws):
                    pos = int(il[ni][ci][i][j])
                    out[ni][ci][pos // ow][pos % ow] = xl[ni][ci][i][j]
    return torch.tensor(out, dtype=x.dtype)


def check_example(solve):
    x = torch.tensor(EX_X)
    y = solve(x, EX_K, stride=EX_S, p=EX_P, mode="lppool2d")
    assert y.shape == (1, 1, 1, 1), f"形状错误: {tuple(y.shape)}"
    assert torch.allclose(y, torch.tensor(EXAMPLE_LPPOOL), atol=1e-6), y.tolist()
    out, idx = solve(x, EX_K, stride=EX_S, mode="maxpool_idx")
    assert torch.allclose(out, torch.tensor(EXAMPLE_OUT), atol=1e-6), out.tolist()
    assert torch.equal(idx, torch.tensor(EXAMPLE_INDICES)), idx.tolist()
    # 手写口径：唯一窗口 [1,2,4,5]
    win = [1.0, 2.0, 4.0, 5.0]
    assert abs(y.item() - sum(v ** 2 for v in win) ** 0.5) < 1e-6, "p=2 应为窗口 L2 范数"
    assert out.item() == 5.0 and idx.item() == 4, "最大值 5 在原图 (1,1)，展平偏移 4"
    # unpool 写回：只有展平位置 4（即 (1,1)）为 5，其余为 0
    up = solve(out, EX_K, stride=EX_S, mode="maxunpool", output_size=(3, 3), indices=idx)
    assert up.shape == (1, 1, 3, 3)
    assert up[0, 0, 1, 1].item() == 5.0
    assert torch.count_nonzero(up).item() == 1


check_example.example = True


def check_lppool2d_matches_native(solve):
    x = _det(2, 3, 7, 8, seed=1)
    # p=2 / 偶数 p 下 abs 口径与原生重合（输入含负值也可对拍）
    for k, s, p in [(2, 2, 2), (3, 1, 2), (2, 3, 2), (3, 3, 2), ((2, 3), (1, 2), 2),
                    (2, 2, 4), (3, 2, 6)]:
        y = solve(x, k, stride=s, p=p, mode="lppool2d")
        assert y.shape == F.lp_pool2d(x, p, k, s).shape, (k, s, p)
        assert torch.allclose(y, F.lp_pool2d(x, p, k, s), atol=1e-6, rtol=1e-6), (k, s, p)
    # 输入非负时 p=1 / 分数 p 也与原生一致（abs 口径与原生 signed 口径重合）
    xn = x.abs() + 0.05
    for k, s, p in [(2, 2, 1), (3, 2, 1.5), (2, 2, 1.0)]:
        y = solve(xn, k, stride=s, p=p, mode="lppool2d")
        assert torch.allclose(y, F.lp_pool2d(xn, p, k, s), atol=1e-6, rtol=1e-6), (k, s, p)


def check_lppool2d_matches_oracle(solve):
    # 含零填充配置（原生 lp_pool2d 无 padding 参数，填充情形与独立 oracle 对拍）
    x = _det(2, 2, 7, 7, seed=2)
    for k, s, p, pad in [(2, 2, 2.0, 0), (3, 1, 2.0, 0), (2, 3, 1.0, 0), (3, 2, 2.0, 1),
                         (2, 1, 1.0, 1), ((3, 2), (2, 1), 2.0, (1, 0))]:
        y = solve(x, k, stride=s, p=p, mode="lppool2d", padding=pad)
        exp = _oracle_lppool2d(x, k, s, p, pad)
        assert y.shape == exp.shape, (k, s, p, pad, y.shape, exp.shape)
        assert torch.allclose(y, exp, atol=1e-5, rtol=1e-5), (k, s, p, pad)


def check_lppool1d(solve):
    x = _det(2, 3, 11, seed=4)
    for k, s in [(2, 2), (3, 1), (4, 2), (3, 3)]:
        y = solve(x, k, stride=s, p=2, mode="lppool1d")
        assert torch.allclose(y, F.lp_pool1d(x, 2, k, s), atol=1e-6, rtol=1e-6), (k, s)
    xn = x.abs() + 0.05
    for k, s, p in [(2, 2, 1), (3, 2, 1.5)]:
        y = solve(xn, k, stride=s, p=p, mode="lppool1d")
        assert torch.allclose(y, F.lp_pool1d(xn, p, k, s), atol=1e-6, rtol=1e-6), (k, s, p)
    # oracle（含 1-D 零填充）
    for k, s, p, pad in [(3, 2, 2.0, 0), (2, 1, 1.0, 1), (3, 3, 2.0, 1)]:
        y = solve(x, k, stride=s, p=p, mode="lppool1d", padding=pad)
        assert torch.allclose(y, _oracle_lppool1d(x, k, s, p, pad), atol=1e-5, rtol=1e-5), \
            (k, s, p, pad)
    # stride=None → kernel
    assert torch.equal(solve(x, 3, mode="lppool1d"), solve(x, 3, stride=3, mode="lppool1d"))


def check_maxpool_idx_matches_native(solve):
    x = _det(2, 3, 7, 5, seed=5)
    for k, s, pad in [(2, 2, 0), (3, 1, 1), (3, 2, 1), (2, 3, 0), ((2, 3), (1, 2), (1, 0))]:
        out, idx = solve(x, k, stride=s, mode="maxpool_idx", padding=pad)
        exp_out, exp_idx = F.max_pool2d(x, k, s, pad, return_indices=True)
        assert torch.allclose(out, exp_out, atol=1e-6, rtol=1e-6), (k, s, pad)
        assert torch.equal(idx, exp_idx), (k, s, pad)
    # 并列最大：取行优先最前者，indices 与原生完全一致
    xq = (x * 3).round() / 3
    for k, s, pad in [(2, 2, 0), (3, 1, 1), (2, 2, 1)]:
        out, idx = solve(xq, k, stride=s, mode="maxpool_idx", padding=pad)
        exp_out, exp_idx = F.max_pool2d(xq, k, s, pad, return_indices=True)
        assert torch.equal(out, exp_out) and torch.equal(idx, exp_idx), (k, s, pad)
    # 独立 oracle 对拍
    for k, s, pad in [(2, 2, 0), (3, 1, 1), (3, 2, 1)]:
        out, idx = solve(x, k, stride=s, mode="maxpool_idx", padding=pad)
        eo, ei = _oracle_maxpool(x, k, s, pad)
        assert torch.allclose(out, eo, atol=1e-6, rtol=1e-6) and torch.equal(idx, ei), (k, s, pad)
    # maxpool_idx 忽略 p
    a = solve(x, 2, stride=2, mode="maxpool_idx")
    b = solve(x, 2, stride=2, p=123.0, mode="maxpool_idx")
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    # stride=None → kernel
    c1 = solve(x, 2, mode="maxpool_idx")
    assert torch.equal(a[0], c1[0]) and torch.equal(a[1], c1[1])


def check_maxunpool_matches_native(solve):
    x = _det(2, 3, 6, 6, seed=6)
    out, idx = solve(x, 2, stride=2, mode="maxpool_idx")
    up = solve(out, 2, stride=2, mode="maxunpool", output_size=(6, 6), indices=idx)
    exp = F.max_unpool2d(out, idx, 2, 2, output_size=(6, 6))
    assert torch.equal(up, exp), "maxunpool 应与 F.max_unpool2d 逐位一致"
    out2, idx2 = F.max_pool2d(x, 3, 3, return_indices=True)
    up2 = solve(out2, 3, stride=3, mode="maxunpool", output_size=(6, 6), indices=idx2)
    assert torch.equal(up2, F.max_unpool2d(out2, idx2, 3, 3, output_size=(6, 6)))
    # output_size=None 按几何关系推断
    up3 = solve(out, 2, stride=2, mode="maxunpool", indices=idx)
    assert torch.equal(up3, up)


def check_maxunpool_roundtrip(solve):
    x = _det(2, 3, 7, 5, seed=7, lo=0.5, hi=1.5)  # 全正，避免 0 值干扰非零位探针
    out, idx = solve(x, 2, stride=2, mode="maxpool_idx")
    up = solve(out, 2, stride=2, mode="maxunpool", output_size=(7, 5), indices=idx)
    assert torch.equal(up, _oracle_unpool(out, idx, (7, 5)))
    # 非零位恰好还原到 indices 指向的源位置（原图 7×5=35 的展平空间）
    flat = up.reshape(2, 3, -1)
    nz = flat != 0
    n_src = x.shape[-2] * x.shape[-1]
    exp_nz = torch.zeros(2, 3, n_src, dtype=torch.bool).scatter(
        2, idx.reshape(2, 3, -1), torch.ones(2, 3, idx.shape[-2] * idx.shape[-1], dtype=torch.bool))
    assert torch.equal(nz, exp_nz), "非零位必须恰为 indices 指向的源位置"
    assert torch.equal(flat.gather(2, idx.reshape(2, 3, -1)), out.reshape(2, 3, -1))


def check_lppool_p1_and_abs_probes(solve):
    # p=1：窗口输出 = |x| 求和（非均值）——刻意与 avg_pool2d 的均值口径区分
    x = torch.tensor([[[[1.0, -2.0], [3.0, -4.0]]]])  # (1,1,2,2) 单窗口
    y = solve(x, 2, stride=2, p=1, mode="lppool2d")
    assert abs(y.item() - 10.0) < 1e-6, y.item()  # 1+2+3+4 = 10
    avg = F.avg_pool2d(x.abs(), 2, 2)
    assert abs(avg.item() - 2.5) < 1e-6 and abs(y.item() - 4 * avg.item()) < 1e-5, \
        "p=1 应为窗口 |x| 求和而非均值"
    # kernel=1：输出 = |x| 逐元素（abs 探针）
    x2 = _det(2, 3, 4, 5, seed=8)
    y2 = solve(x2, 1, stride=1, p=2, mode="lppool2d")
    assert torch.allclose(y2, x2.abs(), atol=1e-6), "kernel=1 时 lppool 应等于逐元素 |x|"
    assert torch.all(y2 >= 0)
    y2b = solve(x2, 1, stride=1, p=5, mode="lppool2d")
    assert torch.allclose(y2b, x2.abs(), atol=1e-5)
    # 1-D 同理
    x3 = _det(1, 2, 6, seed=9)
    y3 = solve(x3, 1, stride=1, p=2, mode="lppool1d")
    assert torch.allclose(y3, x3.abs(), atol=1e-6)


def check_zero_window(solve):
    x = torch.zeros(1, 1, 4, 4)
    x[0, 0, 0, 0] = 5.0
    y = solve(x, 2, stride=2, p=2, mode="lppool2d")
    assert y[0, 0, 0, 0].item() == 5.0
    for i in (0, 1):
        for j in (0, 1):
            if (i, j) != (0, 0):
                assert y[0, 0, i, j].item() == 0.0, "全零窗口输出必须为 0"
    # 零填充产生的全零窗口
    x3 = torch.zeros(1, 1, 3, 3)
    y3 = solve(x3, 2, stride=1, p=2, mode="lppool2d", padding=1)
    assert torch.all(y3 == 0), "全零输入（含零填充）的每个窗口输出都应为 0"
    # 1-D：窗口 0 = [3,0] → 3；窗口 1 = [0,0] → 0
    x4 = torch.zeros(1, 1, 5)
    x4[0, 0, 0] = 3.0
    y4 = solve(x4, 2, stride=2, p=1, mode="lppool1d")
    assert y4.shape == (1, 1, 2)
    assert y4[0, 0, 0].item() == 3.0 and y4[0, 0, 1].item() == 0.0


def check_float64(solve):
    x = _det(1, 2, 4, 4, seed=12).to(torch.float64)
    y = solve(x, 2, stride=2, p=2, mode="lppool2d")
    assert y.dtype == torch.float64
    assert torch.allclose(y, F.lp_pool2d(x, 2, 2, 2), atol=1e-12, rtol=1e-12)
    out, idx = solve(x, 2, stride=2, mode="maxpool_idx")
    eout, eidx = F.max_pool2d(x, 2, 2, return_indices=True)
    assert torch.equal(out, eout) and torch.equal(idx, eidx)


def check_value_errors(solve):
    x4 = _det(2, 3, 6, 6, seed=10)
    x3 = _det(2, 3, 8, seed=11)

    def expect_ve(fn):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError("应抛 ValueError")

    expect_ve(lambda: solve(x4, 2, p=0.5, mode="lppool2d"))  # p < 1
    expect_ve(lambda: solve(x4, 2, p=0, mode="lppool1d"))
    expect_ve(lambda: solve(x4, 2, p=float("inf"), mode="lppool2d"))  # 非有限 p
    expect_ve(lambda: solve(x4, 2, p=float("nan"), mode="lppool1d"))
    expect_ve(lambda: solve(x4, 2, mode="avgpool"))  # 非法 mode
    expect_ve(lambda: solve(x4, 2, mode="lppool3d"))
    expect_ve(lambda: solve(x4, 7, mode="lppool2d"))  # 窗口放不进
    expect_ve(lambda: solve(x4, 2, mode="lppool2d", padding=2))  # padding > kernel/2
    expect_ve(lambda: solve(x3, 2, mode="lppool2d"))  # lppool2d 需要 4 维
    expect_ve(lambda: solve(x4, 2, mode="lppool1d"))  # lppool1d 需要 3 维
    expect_ve(lambda: solve(x3, (2, 3), mode="lppool1d"))  # 1-D kernel/stride/padding 须逐维一致
    expect_ve(lambda: solve(x4, 2, stride=0, mode="lppool2d"))  # stride 须正
    expect_ve(lambda: solve(x4, 0, mode="maxpool_idx"))  # kernel 须正
    out, idx = solve(x4, 2, stride=2, mode="maxpool_idx")
    expect_ve(lambda: solve(out, 2, stride=1, mode="maxunpool", output_size=(6, 6),
                            indices=idx))  # 重叠配置 stride < kernel
    expect_ve(lambda: solve(out, 2, stride=2, mode="maxunpool", output_size=(6, 6),
                            indices=idx, padding=1))  # maxunpool 要求 padding==0
    expect_ve(lambda: solve(out, 2, stride=2, mode="maxunpool", output_size=(6, 6)))  # 缺 indices
    expect_ve(lambda: solve(out, 2, stride=2, mode="maxunpool", output_size=(6, 6),
                            indices=idx[:, :, :2, :]))  # indices 形状不符
    bad = idx.clone()
    bad[0, 0, 0, 0] = 999
    expect_ve(lambda: solve(out, 2, stride=2, mode="maxunpool", output_size=(6, 6),
                            indices=bad))  # indices 越界
    neg = idx.clone()
    neg[0, 0, 0, 0] = -1
    expect_ve(lambda: solve(out, 2, stride=2, mode="maxunpool", output_size=(6, 6),
                            indices=neg))
    expect_ve(lambda: solve(out, 2, stride=2, mode="maxunpool", output_size=(6, 6),
                            indices=idx.to(torch.float32)))  # indices 须整数 dtype
    expect_ve(lambda: solve(out, 2, stride=2, mode="maxunpool", output_size=(2, 2),
                            indices=idx))  # output_size 过小
