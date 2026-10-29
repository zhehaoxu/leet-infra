import torch


def _oracle(x, W, y, lam, eta):
    xs = x.tolist()
    ws = W.tolist()
    ys = y.tolist()
    m, n = len(ws), len(ws[0])
    r = [sum(ws[i][j] * xs[j] for j in range(n)) - ys[i] for i in range(m)]
    g = [sum(ws[i][j] * r[i] for i in range(m)) for j in range(n)]
    z = [xs[j] - eta * g[j] for j in range(n)]
    out = []
    for v in z:
        mag = abs(v) - lam
        out.append(0.0 if mag <= 0 else (mag if v > 0 else -mag))
    sup = [j for j, v in enumerate(out) if v != 0.0]
    return out, sup, z


X = torch.tensor([1.0, -2.0, 0.5])
WT = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
YT = torch.tensor([1.0, 1.0])


def check_example(solve):
    x_new, sup = solve(X, WT, YT, 0.2, 0.5)
    assert torch.allclose(
        x_new, torch.tensor([0.800000011920929, -0.550000011920929, 1.5499999523162842]),
        atol=1e-6,
    ), f"got {x_new.tolist()}"
    assert sup.tolist() == [0, 1, 2], sup.tolist()


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(21)
    for m, n in [(4, 6), (8, 3), (2, 2)]:
        W = torch.randn(m, n)
        x = torch.randn(n)
        y = torch.randn(m)
        for lam, eta in [(0.1, 0.1), (0.3, 0.5), (0.0, 0.2)]:
            x_new, sup = solve(x, W, y, lam, eta)
            exp, exp_sup, _ = _oracle(x, W, y, lam, eta)
            assert torch.allclose(x_new, torch.tensor(exp), atol=1e-5), f"{m}x{n} lam={lam}"
            assert sup.tolist() == exp_sup, f"{m}x{n} lam={lam} sup"


def check_lam_zero_is_gradient_step(solve):
    torch.manual_seed(22)
    W = torch.randn(5, 4)
    x = torch.randn(4)
    y = torch.randn(5)
    eta = 0.1
    x_new, _ = solve(x, W, y, 0.0, eta)
    g = W.transpose(0, 1) @ (W @ x - y)
    assert torch.allclose(x_new, x - eta * g, atol=1e-6), "lam=0 时应恰好退化为梯度步"


def check_large_lam_zeroes_everything(solve):
    torch.manual_seed(23)
    W = torch.randn(3, 5)
    x = torch.randn(5)
    y = torch.randn(3)
    _, _, z = _oracle(x, W, y, 0.0, 0.1)
    lam = max(abs(v) for v in z) + 0.5
    x_new, sup = solve(x, W, y, lam, 0.1)
    assert torch.all(x_new == 0), x_new.tolist()
    assert sup.numel() == 0, sup.tolist()


def check_shrinkage_keeps_sign(solve):
    torch.manual_seed(24)
    W = torch.randn(4, 4)
    x = torch.randn(4)
    y = torch.randn(4)
    lam, eta = 0.15, 0.3
    x_new, _ = solve(x, W, y, lam, eta)
    _, _, z = _oracle(x, W, y, lam, eta)
    for j in range(4):
        if x_new[j] == 0:
            continue
        zj = z[j]
        assert (x_new[j] > 0) == (zj > 0), f"索引 {j} 符号应保持"
        assert abs(abs(float(x_new[j])) - (abs(zj) - lam)) < 1e-5, f"索引 {j} 收缩量应为 lam"


def check_support_is_threshold_set(solve):
    torch.manual_seed(25)
    W = torch.randn(6, 5)
    x = torch.randn(5)
    y = torch.randn(6)
    lam, eta = 0.2, 0.4
    _, sup = solve(x, W, y, lam, eta)
    _, _, z = _oracle(x, W, y, lam, eta)
    expected = [j for j in range(5) if abs(z[j]) > lam]
    assert sup.tolist() == expected, (sup.tolist(), expected)


def check_eta_zero_only_shrinks(solve):
    x = torch.tensor([2.0, -1.0, 0.3])
    W = torch.zeros(2, 3)
    y = torch.zeros(2)
    x_new, sup = solve(x, W, y, 0.5, 1.0)
    assert torch.allclose(x_new, torch.tensor([1.5, -0.5, 0.0]), atol=1e-6), x_new.tolist()
    assert sup.tolist() == [0, 1], sup.tolist()


def check_threshold_equality_zeroed(solve):
    # |z| 恰好等于 lam 时应置 0（严格大于才保留）
    x = torch.tensor([0.1, -0.1])
    W = torch.eye(2)
    y = torch.zeros(2)
    x_new, sup = solve(x, W, y, 0.1, 0.0)
    assert torch.all(x_new == 0), x_new.tolist()
    assert sup.numel() == 0, sup.tolist()
