import torch


def _oracle(X, W, alphas, bits):
    """独立 oracle：纯 Python 逐 token / 逐通道量化 + 逐元素点积，float64 计算。"""
    Xl = X.to(torch.float64).tolist()
    Wl = W.to(torch.float64).tolist()
    T, C = len(Xl), len(Xl[0])
    O = len(Wl)
    qmax = 2 ** (bits - 1) - 1
    Y = [
        [sum(Xl[i][j] * Wl[o][j] for j in range(C)) for o in range(O)]
        for i in range(T)
    ]
    sx = [max(abs(Xl[i][j]) for i in range(T)) for j in range(C)]
    sw = [max(abs(Wl[o][j]) for o in range(O)) for j in range(C)]

    mses = []
    for a in alphas:
        a = float(a)
        s = [1.0 if sx[j] == 0.0 or sw[j] == 0.0 else sx[j] ** a / sw[j] ** (1.0 - a) for j in range(C)]
        Xs = [[Xl[i][j] / s[j] for j in range(C)] for i in range(T)]
        Ws = [[Wl[o][j] * s[j] for j in range(C)] for o in range(O)]
        Xd = []
        for i in range(T):
            amax = max(abs(v) for v in Xs[i])
            Xd.append([0.0] * C if amax == 0.0 else
                      [max(-qmax, min(qmax, round(v / (amax / qmax)))) * (amax / qmax) for v in Xs[i]])
        Wd = []
        for o in range(O):
            amax = max(abs(v) for v in Ws[o])
            Wd.append([0.0] * C if amax == 0.0 else
                      [max(-qmax, min(qmax, round(v / (amax / qmax)))) * (amax / qmax) for v in Ws[o]])
        se = 0.0
        for i in range(T):
            for o in range(O):
                se += (Y[i][o] - sum(Xd[i][j] * Wd[o][j] for j in range(C))) ** 2
        mses.append(se / (T * O))

    best = 0
    for i in range(1, len(mses)):
        if mses[i] < mses[best]:
            best = i
    sx_l, sw_l = sx, sw
    a = float(alphas[best])
    s = [1.0 if sx_l[j] == 0.0 or sw_l[j] == 0.0 else sx_l[j] ** a / sw_l[j] ** (1.0 - a) for j in range(C)]
    return float(alphas[best]), torch.tensor(s, dtype=torch.float64), torch.tensor(mses, dtype=torch.float64)


def check_example(solve):
    X = torch.tensor(
        [[0.9, 1.1, 55.0], [1.2, 0.7, 62.0], [-0.8, 1.4, 70.0]], dtype=torch.float32
    )
    W = torch.tensor([[0.4, -0.6, 0.05], [-0.5, 0.3, -0.04]], dtype=torch.float32)
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    best, scales, mses = solve(X, W, alphas, 8)
    assert abs(best - 0.5) < 1e-9, f"best_alpha 应为 0.5，got {best}"
    assert torch.allclose(
        scales, torch.tensor([1.549193, 1.527525, 37.416573]), atol=1e-5
    ), f"got {scales.tolist()}"
    exp = torch.tensor([1.94763e-4, 1.61371e-4, 9.45225e-6, 3.95171e-5, 1.90658e-4])
    assert torch.allclose(mses, exp, rtol=1e-4, atol=1e-10), f"got {mses.tolist()}"


check_example.example = True


def check_uniform_exact_tie(solve):
    """全 1 输入可被整数量化精确表示：所有 alpha 的 MSE 相同，取网格中先出现者。"""
    X = torch.ones(2, 3)
    W = torch.ones(2, 3)
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    best, scales, mses = solve(X, W, alphas, 8)
    assert best == 0.0, f"并列应取先出现的 0.0，got {best}"
    assert float(mses.max() - mses.min()) == 0.0, f"MSE 应完全相等，got {mses.tolist()}"
    assert torch.allclose(scales, torch.ones(3)), f"got {scales.tolist()}"


def check_activation_outlier(solve):
    """激活有离群通道时，把一部分量迁到权重能显著降误差，非零 alpha 胜出。"""
    X = torch.tensor(
        [[1.0, 0.5, 80.0], [0.8, 1.2, 60.0], [-1.1, 0.9, 70.0], [0.6, -1.3, 90.0]]
    )
    W = torch.tensor([[0.5, -0.4, 0.02], [-0.3, 0.6, 0.01], [0.7, 0.2, -0.03]])
    best, _, mses = solve(X, W, [0.0, 0.25, 0.5, 0.75, 1.0], 8)
    assert best > 0.0, f"离群激活下 alpha=0 不该胜出，got {best}"
    assert float(mses[2]) < float(mses[0]), f"alpha=0.5 应优于 alpha=0，got {mses.tolist()}"


def check_weight_outlier_low_alpha(solve):
    """权重侧有离群通道时，低 alpha（把量推给激活）更优。"""
    X = torch.tensor([[1.0, 1.2, 0.8], [0.9, 1.1, 1.3], [1.1, 0.7, 1.0]])
    W = torch.tensor([[0.2, 0.3, 9.0], [-0.4, 0.1, 7.0], [0.3, -0.2, 8.0]])
    best, _, mses = solve(X, W, [0.0, 0.25, 0.5, 0.75, 1.0], 8)
    assert abs(best - 0.25) < 1e-9, f"got {best}"
    assert float(mses[1]) < float(mses[0]), f"got {mses.tolist()}"


def check_zero_channels(solve):
    """激活与权重同时为零的通道：s 取 1，不产生 NaN/Inf。"""
    X = torch.tensor([[0.0, 1.0, 2.0], [0.0, 3.0, 4.0]])
    W = torch.tensor([[0.0, 1.0, 2.0], [1.0, 0.5, 0.25]])
    best, scales, mses = solve(X, W, [0.0, 0.5, 1.0], 8)
    assert torch.isfinite(mses).all(), f"got {mses.tolist()}"
    assert torch.isfinite(scales).all(), f"got {scales.tolist()}"
    assert scales[0].item() == 1.0, f"零通道 s 应为 1，got {scales.tolist()}"
    assert abs(best - 0.5) < 1e-9, f"got {best}"


def check_zero_activation_row(solve):
    """整行为 0 的 token：反量化后仍为 0，误差只来自其它行。"""
    torch.manual_seed(5)
    X = torch.rand(4, 5, dtype=torch.float32) * 4
    X[2] = 0.0
    W = torch.randn(3, 5, dtype=torch.float32)
    best, scales, mses = solve(X, W, [0.0, 0.5, 1.0], 8)
    exp_best, exp_s, exp_m = _oracle(X, W, [0.0, 0.5, 1.0], 8)
    assert best == exp_best, f"got {best} vs {exp_best}"
    assert torch.allclose(scales, exp_s.to(torch.float32), atol=1e-5)
    assert torch.allclose(mses, exp_m.to(torch.float32), rtol=1e-4, atol=1e-10)
    assert torch.isfinite(mses).all()


def check_bits_four(solve):
    """位宽变化只改 qmax，整条搜索链路仍成立。"""
    torch.manual_seed(9)
    X = torch.rand(3, 4, dtype=torch.float32) + 0.1
    W = torch.randn(2, 4, dtype=torch.float32)
    alphas = [0.0, 0.5, 1.0]
    best, scales, mses = solve(X, W, alphas, 4)
    exp_best, exp_s, exp_m = _oracle(X, W, alphas, 4)
    assert best == exp_best, f"got {best} vs {exp_best}"
    assert torch.allclose(scales, exp_s.to(torch.float32), atol=1e-5)
    assert torch.allclose(mses, exp_m.to(torch.float32), rtol=1e-4, atol=1e-10)


def check_bad_bits(solve):
    X = torch.ones(2, 2)
    W = torch.ones(2, 2)
    for bits in (1, 9):
        try:
            solve(X, W, [0.5], bits)
        except ValueError:
            continue
        raise AssertionError(f"bits={bits} 应抛 ValueError")


def check_scale_of_best(solve):
    """返回的 scales 必须是所选 alpha 下的 s，而不是其它 alpha 的。"""
    torch.manual_seed(21)
    X = torch.rand(3, 4, dtype=torch.float32) * 3
    X[:, 2] *= 20
    W = torch.randn(3, 4, dtype=torch.float32)
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    best, scales, mses = solve(X, W, alphas, 8)
    sx = X.to(torch.float64).abs().amax(0)
    sw = W.to(torch.float64).abs().amax(0)
    a = float(best)
    exp = sx ** a / sw ** (1.0 - a)
    assert torch.allclose(scales, exp.to(torch.float32), atol=1e-5), f"got {scales.tolist()} vs {exp.tolist()}"


def check_random_against_oracle(solve):
    """随机校准集 / 权重 / alpha 网格上与独立 oracle 一致。"""
    g = torch.Generator().manual_seed(20261029)
    for _ in range(60):
        T = int(torch.randint(1, 6, (1,), generator=g))
        C = int(torch.randint(1, 7, (1,), generator=g))
        O = int(torch.randint(1, 5, (1,), generator=g))
        X = torch.randn(T, C, generator=g, dtype=torch.float32)
        W = torch.randn(O, C, generator=g, dtype=torch.float32)
        if C >= 2:
            X[:, int(torch.randint(0, C, (1,), generator=g))] *= 10.0
        alphas = sorted(set(float(torch.randint(0, 5, (1,), generator=g)) / 4.0 for _ in range(5)))
        bits = int(torch.randint(4, 9, (1,), generator=g))
        got_best, got_s, got_m = solve(X, W, alphas, bits)
        exp_best, exp_s, exp_m = _oracle(X, W, alphas, bits)
        tag = (T, C, O, alphas, bits)
        assert got_best == exp_best, f"{tag}: {got_best} vs {exp_best}"
        assert torch.allclose(got_s, exp_s.to(torch.float32), atol=1e-5), f"{tag}: scales 不符"
        assert torch.allclose(got_m, exp_m.to(torch.float32), rtol=1e-4, atol=1e-10), f"{tag}: mses 不符"
        assert torch.isfinite(got_m).all(), f"{tag}: 出现非有限 MSE"
