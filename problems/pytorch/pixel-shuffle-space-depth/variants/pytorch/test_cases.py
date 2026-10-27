import torch
import torch.nn.functional as F

# 示例：手写 (1, 4, 2, 2) 小张量（四个 2×2 通道块），r=2
EXAMPLE_X = [[[[1.0, 2.0], [3.0, 4.0]],
              [[5.0, 6.0], [7.0, 8.0]],
              [[9.0, 10.0], [11.0, 12.0]],
              [[13.0, 14.0], [15.0, 16.0]]]]
# shuffle 后的 (1, 1, 4, 4)（实测回填，参考实现 = 语义同 F.pixel_shuffle）
EXAMPLE_SHUFFLED = [[[[1.0, 5.0, 2.0, 6.0], [9.0, 13.0, 10.0, 14.0],
                      [3.0, 7.0, 4.0, 8.0], [11.0, 15.0, 12.0, 16.0]]]]


def _det(*shape, seed=0, lo=-1.0, hi=1.0):
    """无 RNG 的确定性伪随机张量：黄金比例哈希，CPU/float32 下完全可复现。"""
    numel = 1
    for d in shape:
        numel *= d
    base = torch.arange(numel, dtype=torch.float64)
    v = ((base + 1.0) * (seed + 1) * 0.6180339887498949) % 1.0
    return (v * (hi - lo) + lo).to(torch.float32).reshape(*shape)


def _oracle_d2s(x, r):
    """独立循环 oracle（d2s/shuffle）：逐目标坐标算来源下标，与向量化 reshape/permute 分路径。"""
    n, cr2, h, w = x.shape
    c = cr2 // (r * r)
    xl = x.tolist()
    out = [[[[None] * (w * r) for _ in range(h * r)] for _ in range(c)] for _ in range(n)]
    for ni in range(n):
        for ci in range(c):
            for ho in range(h * r):
                for wo in range(w * r):
                    i, j = ho % r, wo % r
                    out[ni][ci][ho][wo] = xl[ni][ci * r * r + i * r + j][ho // r][wo // r]
    return torch.tensor(out, dtype=x.dtype)


def _oracle_s2d(x, r):
    """独立循环 oracle（s2d/unshuffle）：逐输出通道与目标坐标算来源下标。"""
    n, c, h, w = x.shape
    hb, wb = h // r, w // r
    xl = x.tolist()
    out = [[[[None] * wb for _ in range(hb)] for _ in range(c * r * r)] for _ in range(n)]
    for ni in range(n):
        for co in range(c * r * r):
            ci = co // (r * r)
            i = (co % (r * r)) // r
            j = co % r
            for hi in range(hb):
                for wi in range(wb):
                    out[ni][co][hi][wi] = xl[ni][ci][hi * r + i][wi * r + j]
    return torch.tensor(out, dtype=x.dtype)


def check_example(solve):
    x = torch.tensor(EXAMPLE_X)
    y = solve(x, 2, mode="shuffle")
    assert y.shape == (1, 1, 4, 4), f"形状错误: {tuple(y.shape)}"
    assert torch.allclose(y, torch.tensor(EXAMPLE_SHUFFLED), atol=1e-6), y.tolist()
    # 手写口径：out[2h+i][2w+j] = ch[i*2+j][h][w]（通道按块内 [i][j] 序交错铺开）
    assert y[0, 0, 0, :].tolist() == [1.0, 5.0, 2.0, 6.0]
    assert y[0, 0, 2, :].tolist() == [3.0, 7.0, 4.0, 8.0]
    # 双射回程
    assert torch.equal(solve(y, 2, mode="unshuffle"), x), "unshuffle(shuffle(x)) 应还原 x"
    assert torch.equal(solve(x, 2, mode="s2d"), solve(x, 2, mode="unshuffle"))
    assert torch.equal(solve(solve(x, 2, mode="s2d"), 2, mode="d2s"), x)


check_example.example = True


def check_matches_native(solve):
    # N=1 与 batch>1、多种 r / C / H / W（H、W 均被 r 整除，四种 mode 同一配置全查）：
    # 全部与原生逐位一致（纯重排，用 torch.equal）
    for n, c, h, w, r in [(1, 4, 4, 6, 2), (2, 8, 4, 4, 2), (3, 2, 2, 2, 1),
                          (2, 18, 6, 6, 3), (1, 9, 6, 3, 3), (2, 4, 6, 8, 2)]:
        x = _det(n, c, h, w, seed=n + c + h + w)
        assert torch.equal(solve(x, r, mode="shuffle"), F.pixel_shuffle(x, r)), (n, c, h, w, r)
        xu = _det(n, c, h * r, w * r, seed=2 * h + 3 * w + r)
        assert torch.equal(solve(xu, r, mode="unshuffle"), F.pixel_unshuffle(xu, r)), (n, c, h, w, r)
        # docstring 声明的等价关系：s2d ≡ F.pixel_unshuffle、d2s ≡ F.pixel_shuffle
        assert torch.equal(solve(x, r, mode="s2d"), F.pixel_unshuffle(x, r)), ("s2d", n, c, h, w, r)
        xd = _det(n, c * r * r, h, w, seed=5 * c + 7 * h + 2 * r)
        assert torch.equal(solve(xd, r, mode="d2s"), F.pixel_shuffle(xd, r)), ("d2s", n, c, h, w, r)


def check_double_loop_oracle(solve):
    # 独立双层（四层）循环 oracle：逐目标坐标算来源下标
    for n, c, h, w, r in [(2, 3, 4, 6, 2), (1, 2, 3, 3, 3), (2, 4, 4, 6, 2), (1, 8, 2, 4, 2)]:
        x = _det(n, c, h, w, seed=10 * n + c)
        assert torch.equal(solve(x, r, mode="s2d"), _oracle_s2d(x, r)), (n, c, h, w, r)
        xd = _det(n, c * r * r, h, w, seed=10 * c + h)
        y = solve(xd, r, mode="d2s")
        assert torch.equal(y, _oracle_d2s(xd, r)), (n, c, h, w, r)
        # 双射：输出是输入值的置换（展平后多重集不变）
        assert torch.equal(y.reshape(-1).sort().values, xd.reshape(-1).sort().values)


def check_roundtrip(solve):
    for n, c, h, w, r in [(1, 4, 4, 6, 2), (2, 9, 6, 6, 3), (3, 8, 2, 2, 2)]:
        x = _det(n, c, h, w, seed=n + h + r)
        u = solve(x, r, mode="unshuffle")
        assert torch.equal(solve(u, r, mode="shuffle"), x), "unshuffle∘shuffle 应为恒等"
        d = solve(x, r, mode="s2d")
        assert torch.equal(solve(d, r, mode="d2s"), x), "d2s∘s2d 应为恒等"


def check_r1_identity(solve):
    for n, c, h, w in [(1, 3, 4, 5), (2, 2, 3, 3)]:
        x = _det(n, c, h, w, seed=20 + n + c)
        for mode in ("shuffle", "unshuffle", "s2d", "d2s"):
            assert torch.equal(solve(x, 1, mode=mode), x), f"r=1 时 {mode} 应严格恒等"


def check_non_contiguous(solve):
    base = _det(2, 8, 8, 10, seed=3)
    xt = base.transpose(-1, -2)  # (2, 8, 10, 8) 非连续视图
    assert not xt.is_contiguous()
    for mode, native in (("shuffle", F.pixel_shuffle), ("unshuffle", F.pixel_unshuffle),
                         ("s2d", F.pixel_unshuffle), ("d2s", F.pixel_shuffle)):
        y = solve(xt, 2, mode=mode)
        yc = solve(xt.contiguous(), 2, mode=mode)
        assert torch.equal(y, yc), f"非连续输入结果须与连续输入一致（{mode}）"
        assert torch.equal(y, native(xt.contiguous(), 2)), mode


def check_value_errors(solve):
    x = _det(1, 4, 4, 4, seed=1)

    def expect_ve(fn):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError("应抛 ValueError")

    expect_ve(lambda: solve(_det(1, 6, 4, 4, seed=2), 2, mode="shuffle"))  # 通道 6 不被 r²=4 整除
    expect_ve(lambda: solve(_det(1, 6, 4, 4, seed=3), 2, mode="d2s"))
    expect_ve(lambda: solve(_det(1, 2, 5, 4, seed=4), 2, mode="unshuffle"))  # H=5 不被 2 整除
    expect_ve(lambda: solve(_det(1, 2, 4, 5, seed=5), 2, mode="s2d"))  # W=5 不被 2 整除
    expect_ve(lambda: solve(x, 2, mode="depth_to_space"))  # 非法 mode
    expect_ve(lambda: solve(x, 0, mode="shuffle"))  # r < 1
    expect_ve(lambda: solve(x, -2, mode="unshuffle"))
    expect_ve(lambda: solve(x, 1.5, mode="shuffle"))  # 非 int
    expect_ve(lambda: solve(x, True, mode="shuffle"))  # bool 不算合法 r
    expect_ve(lambda: solve(torch.zeros(4, 4, 4), 2, mode="shuffle"))  # 非 4 维
