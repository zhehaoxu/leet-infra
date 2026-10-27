import math

import torch


def _oracle(x0, noise, alpha_bar):
    """独立 oracle：逐元素展开成 a·x + s·ε 两个乘积之和。"""
    a = math.sqrt(alpha_bar)
    s = math.sqrt(1.0 - alpha_bar)
    out = []
    for xv, ev in zip(x0.flatten().tolist(), noise.flatten().tolist()):
        out.append(a * xv + s * ev)
    return torch.tensor(out).reshape(x0.shape)


def check_example(solve):
    alpha_bars = torch.tensor([1.0, 0.81, 0.64, 0.49, 0.36])
    x0 = torch.tensor([1.0, -2.0])
    eps = torch.tensor([0.5, 1.0])

    got, idx = solve(x0, eps, alpha_bars, 0.5)
    assert idx == 2, f"strength=0.5、T=5 时起点应为 floor(0.5·4)=2，got {idx}"
    assert [round(float(e), 6) for e in got] == [1.1, -1.0], got.tolist()
    assert got.dtype == torch.float32 and got.shape == x0.shape

    top, top_idx = solve(x0, eps, alpha_bars, 1.0)
    assert top_idx == 4 and [round(float(e), 6) for e in top] == [1.0, -0.4], top.tolist()


check_example.example = True


def check_strength_bounds(solve):
    alpha_bars = torch.tensor([1.0, 0.81, 0.64, 0.49, 0.36])
    x0 = torch.tensor([[0.3, -1.5], [2.0, 0.7]])
    eps = torch.tensor([[0.2, 0.9], [-0.4, 1.3]])

    # strength=0：起点是最干净的一档；ᾱ₀=1 时输出就是 x0 本身
    zero, idx0 = solve(x0, eps, alpha_bars, 0.0)
    assert idx0 == 0 and torch.allclose(zero, x0.to(torch.float32), atol=1e-7), "s=0 应原样返回 x0"

    # strength=1：起点是最后一档（最噪声）
    full, idx_full = solve(x0, eps, alpha_bars, 1.0)
    assert idx_full == len(alpha_bars) - 1
    want_full = _oracle(x0, eps, float(alpha_bars[-1]))
    assert torch.allclose(full, want_full.to(torch.float32), atol=1e-6)

    # 索引随 strength 单调不减，且始终落在 [0, T-1]
    idxs = [solve(x0, eps, alpha_bars, s / 20.0)[1] for s in range(21)]
    assert idxs == sorted(idxs), f"索引必须随 strength 单调不减：{idxs}"
    assert all(0 <= i <= len(alpha_bars) - 1 for i in idxs)


def check_roundtrip(solve):
    """给定第 k 档的 x_t 与同一 ε，按 s=k/(T-1) 应精确还原该 x_t。"""
    alpha_bars = torch.linspace(1.0, 0.05, 5, dtype=torch.float64)
    torch.manual_seed(0)
    x0 = torch.randn(2, 3, dtype=torch.float64)
    eps = torch.randn(2, 3, dtype=torch.float64)

    for k in (1, 2, 3):
        ab = float(alpha_bars[k])
        x_t = math.sqrt(ab) * x0 + math.sqrt(1.0 - ab) * eps
        noise = (x_t - math.sqrt(ab) * x0) / math.sqrt(1.0 - ab)
        got, idx = solve(x0, noise, alpha_bars, k / 4.0)
        assert idx == k, f"k={k} 时起点索引应为 {k}，got {idx}"
        assert torch.allclose(got, x_t.to(torch.float32), atol=1e-6), f"k={k} 未还原到该档"


def check_forward_identity(solve):
    alpha_bars = torch.tensor([1.0, 0.81, 0.64, 0.49, 0.36])
    torch.manual_seed(1)
    x0 = torch.randn(3, 4, dtype=torch.float64)
    eps = torch.randn(3, 4, dtype=torch.float64)

    for s in (0.0, 0.3, 0.6, 1.0):
        got, idx = solve(x0, eps, alpha_bars, s)
        ab = float(alpha_bars[idx])
        a, sg = math.sqrt(ab), math.sqrt(1.0 - ab)
        assert abs(a * a + sg * sg - 1.0) < 1e-12
        # 反解出 ε：应与入参 ε 完全一致（系数对才可能成立）
        back = (got.to(torch.float64) - a * x0) / sg if sg > 0 else None
        if back is not None:
            assert torch.allclose(back, eps, atol=1e-5), f"s={s} 时系数与 ᾱ_i 不匹配"
        assert torch.allclose(got, (a * x0 + sg * eps).to(torch.float32), atol=1e-6)

    # 只有两档时 strength=0.5 也应落在第 0 档
    _, tiny_idx = solve(x0, eps, torch.tensor([0.9, 0.1]), 0.5)
    assert tiny_idx == 0


def check_oracle_and_invalid(solve):
    alpha_bars = torch.linspace(0.99, 0.01, 17, dtype=torch.float64)
    torch.manual_seed(2)
    x0 = torch.randn(2, 2, 2, dtype=torch.float64)
    eps = torch.randn(2, 2, 2, dtype=torch.float64)

    for s in (0.0, 0.125, 0.4375, 0.75, 1.0):
        got, idx = solve(x0, eps, alpha_bars, s)
        want = _oracle(x0, eps, float(alpha_bars[idx]))
        assert torch.allclose(got, want.to(torch.float32), atol=1e-6), f"s={s} 与 oracle 不一致"
        assert idx == int(math.floor(s * 16)), f"s={s} 的索引应为 floor(s·(T-1))"

    good_x = torch.zeros(2)
    good_e = torch.ones(2)
    good_ab = torch.tensor([1.0, 0.5, 0.1])
    for bad in (
        lambda: solve([0.0, 1.0], good_e, good_ab, 0.5),
        lambda: solve(good_x, torch.ones(3), good_ab, 0.5),
        lambda: solve(good_x, good_e, [1.0, 0.5], 0.5),
        lambda: solve(good_x, good_e, torch.tensor([[1.0, 0.5]]), 0.5),
        lambda: solve(good_x, good_e, torch.tensor([1.0]), 0.5),
        lambda: solve(good_x, good_e, torch.tensor([0.5, 0.6, 0.1]), 0.5),
        lambda: solve(good_x, good_e, torch.tensor([1.0, 0.5, 0.5]), 0.5),
        lambda: solve(good_x, good_e, torch.tensor([1.0, 0.5, 0.0]), 0.5),
        lambda: solve(good_x, good_e, torch.tensor([1.0, 0.5, 1.2]), 0.5),
        lambda: solve(good_x, good_e, torch.tensor([1.0, 0.5, float("nan")]), 0.5),
        lambda: solve(good_x, good_e, good_ab, -0.1),
        lambda: solve(good_x, good_e, good_ab, 1.1),
        lambda: solve(good_x, good_e, good_ab, float("nan")),
    ):
        try:
            bad()
            raise AssertionError("非法输入应抛 ValueError")
        except ValueError:
            pass
