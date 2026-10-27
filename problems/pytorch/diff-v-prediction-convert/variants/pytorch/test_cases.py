import math

import torch


def _oracle(x_t, alpha_bar, pred, mode):
    """独立 oracle：逐元素用 math 重算，公式展开成不共享中间量的两支。"""
    a = math.sqrt(alpha_bar)
    s = math.sqrt(1.0 - alpha_bar)
    out = []
    for xt, p in zip(x_t.flatten().tolist(), pred.flatten().tolist()):
        if mode in ("eps", "epsilon"):
            eps = p
            x0 = xt / a - (s / a) * p
            v = a * p - s * x0
        elif mode == "x0":
            x0 = p
            eps = xt / s - (a / s) * p
            v = a * eps - s * p
        else:
            v = p
            eps = a * p + s * xt
            x0 = a * xt - s * p
        out.append((x0, eps, v))
    shape = tuple(x_t.shape)
    return tuple(torch.tensor([row[i] for row in out]).reshape(shape) for i in range(3))


_A = 0.64  # ᾱ = 0.64 → α = 0.8, σ = 0.6
_X = [0.5, -1.0, 2.0]


def check_example(solve):
    x_t = torch.tensor(_X)
    x0, eps, v = solve(x_t, _A, torch.tensor([0.3, 0.0, -0.9]), "eps")
    # x0 = (x_t - 0.6 * eps) / 0.8
    want_x0 = torch.tensor([(0.5 - 0.6 * 0.3) / 0.8, -1.0 / 0.8, (2.0 + 0.6 * 0.9) / 0.8])
    assert torch.allclose(x0, want_x0, atol=1e-6), x0.tolist()
    # v = 0.8 * eps - 0.6 * x0
    want_v = 0.8 * torch.tensor([0.3, 0.0, -0.9]) - 0.6 * want_x0
    assert torch.allclose(v, want_v, atol=1e-6), v.tolist()
    assert torch.allclose(eps, torch.tensor([0.3, 0.0, -0.9]), atol=1e-6), "已知项应原样返回"
    assert x0.dtype == torch.float32


check_example.example = True


def check_forward_consistency(solve):
    """三条关系必须同时成立：x_t = α x0 + σ eps、v = α eps - σ x0、三种 mode 互逆。"""
    torch.manual_seed(0)
    x_t = torch.randn(4, 5, dtype=torch.float64)
    for ab in (0.05, 0.3, 0.64, 0.9, 0.99):
        a, s = math.sqrt(ab), math.sqrt(1.0 - ab)
        x0, eps, v = solve(x_t, ab, torch.randn(4, 5, dtype=torch.float64), "eps")
        assert torch.allclose(a * x0 + s * eps, x_t.to(torch.float32), atol=1e-5), "x_t 重建失败"
        assert torch.allclose(v, a * eps - s * x0, atol=1e-5), "v 关系不成立"

        # 把上一步的 x0 再当作已知量反解，应回到同一个 eps
        _, eps2, _ = solve(x_t, ab, x0, "x0")
        assert torch.allclose(eps2, eps, atol=1e-5), "x0 -> eps 未与 eps -> x0 互逆"

        # v -> eps / v -> x0 同样互逆
        x0_3, eps_3, _ = solve(x_t, ab, v, "v")
        assert torch.allclose(x0_3, x0, atol=1e-5) and torch.allclose(eps_3, eps, atol=1e-5)


def check_degenerate(solve):
    """ᾱ → 0 时 x_t 全是噪声；ᾱ → 1 时 x_t 就是 x0。"""
    torch.manual_seed(1)
    noise = torch.randn(3, 4, dtype=torch.float64)

    # ᾱ = 1（仅 eps mode 合法）：x0 = x_t，v = eps（因为 σ = 0）
    x0, eps, v = solve(noise, 1.0, noise, "eps")
    assert torch.allclose(x0, noise.to(torch.float32), atol=1e-6) and torch.allclose(v, eps, atol=1e-6)

    # ᾱ 很小：x0 幅度被放大 1/α，用关系式校验而不是直接比数值
    ab = 1e-4
    x0, eps, v = solve(noise, ab, noise, "eps")
    a, s = math.sqrt(ab), math.sqrt(1.0 - ab)
    assert torch.allclose(a * x0 + s * eps, noise.to(torch.float32), atol=1e-5)

    # v mode 在 ᾱ = 0 与 1 两端都合法
    for ab_edge in (0.0, 1.0):
        x0, eps, v = solve(torch.full((2, 2), 0.5), ab_edge, torch.full((2, 2), 0.25), "v")
        assert torch.isfinite(x0).all() and torch.isfinite(eps).all()


def check_random_vs_oracle(solve):
    torch.manual_seed(2)
    x_t = torch.randn(3, 2, 3, dtype=torch.float64)
    pred = torch.randn(3, 2, 3, dtype=torch.float64) * 2.0
    ab = 0.36
    for mode in ("eps", "x0", "v"):
        got = solve(x_t, ab, pred, mode)
        want = _oracle(x_t, ab, pred, mode)
        for got_i, want_i in zip(got, want):
            assert torch.allclose(got_i, want_i.to(torch.float32), atol=1e-5), f"{mode} 与 oracle 不一致"


def check_invalid_inputs(solve):
    good = torch.zeros(2, 2)
    for bad in (
        lambda: solve([0.0, 1.0], 0.5, good, "eps"),
        lambda: solve(good, 0.5, torch.zeros(4), "eps"),
        lambda: solve(good, 0.5, good, "velocity"),
        lambda: solve(good, -0.1, good, "eps"),
        lambda: solve(good, 1.2, good, "eps"),
        lambda: solve(good, 0.0, good, "eps"),
        lambda: solve(good, 1.0, good, "x0"),
    ):
        try:
            bad()
            raise AssertionError("非法输入应抛 ValueError")
        except ValueError:
            pass
