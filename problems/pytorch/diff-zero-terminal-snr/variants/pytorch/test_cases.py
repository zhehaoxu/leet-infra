import math

import torch


def _oracle(betas):
    """独立 oracle：python 循环滚累积乘积 + math 逐个重算（不共享任何中间张量）。"""
    raw = []
    acc = 1.0
    for b in betas.tolist():
        acc *= 1.0 - b
        raw.append(acc)
    last = raw[-1]
    ab, lam, ang = [], [], []
    for r in raw:
        v = (r - last) / (1.0 - last)
        v = min(max(v, 0.0), 1.0)
        ab.append(v)
        if v <= 0.0:
            lam.append(float("-inf"))
            ang.append(math.pi / 2.0)
        elif v >= 1.0:
            lam.append(float("inf"))
            ang.append(0.0)
        else:
            lam.append(math.log(v / (1.0 - v)))
            ang.append(math.atan(math.sqrt((1.0 - v) / v)))
    return torch.tensor(ab), torch.tensor(lam), torch.tensor(ang)


def check_example(solve):
    alpha_bars, log_snrs, angles = solve(torch.tensor([0.1, 0.2, 0.4]))

    want_ab = [0.823944, 0.507042, 0.0]
    want_lam = [1.543298, 0.028171]
    want_ang = [0.432994, 0.778356, 1.570796]

    got_ab = [round(float(e), 6) for e in alpha_bars]
    assert got_ab == want_ab, f"重标定后的 ᾱ 不对：{got_ab}"
    got_lam = [round(float(e), 6) for e in log_snrs[:2]]
    assert got_lam == want_lam, f"log-SNR 不对：{got_lam}"
    got_ang = [round(float(e), 6) for e in angles]
    assert got_ang == want_ang, f"旋转角不对：{got_ang}"
    assert alpha_bars.dtype == torch.float32


check_example.example = True


def check_terminal_zero_snr(solve):
    gen = torch.Generator().manual_seed(0)
    for n in (2, 5, 50):
        betas = torch.rand(n, generator=gen, dtype=torch.float64) * 0.6
        alpha_bars, log_snrs, angles = solve(betas)
        assert float(alpha_bars[-1]) == 0.0, "末端 ᾱ 必须精确为 0（零终端 SNR）"
        assert float(log_snrs[-1]) == float("-inf"), "末端 λ 必须是 -inf"
        assert abs(float(angles[-1]) - math.pi / 2.0) < 1e-6, "末端应为 90° 旋转（v 完全转向 -x₀）"
        # 末端之前都还有信号，不能被重标定一起压成 0
        assert float(alpha_bars[-2]) > 0.0, "只有末位可以为零"


def check_clean_end_rotation(solve):
    alpha_bars, log_snrs, angles = solve(torch.tensor([0.0, 0.3, 0.5]))
    assert float(alpha_bars[0]) == 1.0, "β₀ = 0 时首项 ᾱ' 必须保持 1"
    assert float(log_snrs[0]) == float("inf"), "首项 λ 应为 +inf"
    assert abs(float(angles[0])) < 1e-7, "首项旋转角应为 0（v = ε）"

    # 旋转角与 α / σ 严格一致：cosθ = α、sinθ = σ
    for ab, ang in zip(alpha_bars.tolist(), angles.tolist()):
        a, s = math.sqrt(ab), math.sqrt(1.0 - ab)
        assert abs(math.cos(ang) - a) < 1e-6 and abs(math.sin(ang) - s) < 1e-6, "角度与 α/σ 不自洽"
        assert abs(a * a + s * s - 1.0) < 1e-7


def check_monotone_and_identities(solve):
    gen = torch.Generator().manual_seed(1)
    betas = torch.rand(20, generator=gen, dtype=torch.float64) * 0.3
    alpha_bars, log_snrs, angles = solve(betas)

    ab = alpha_bars.tolist()
    lam = log_snrs.tolist()
    ang = angles.tolist()
    assert all(ab[i] >= ab[i + 1] for i in range(len(ab) - 1)), "ᾱ 必须单调不增"
    assert all(lam[i] >= lam[i + 1] for i in range(len(lam) - 1)), "λ 必须单调不增"
    assert all(ang[i] <= ang[i + 1] for i in range(len(ang) - 1)), "θ 必须单调不减"

    for v, l, th in zip(ab[:-1], lam[:-1], ang[:-1]):
        assert abs(l - math.log(v / (1.0 - v))) < 1e-4, "λ 与 ᾱ 不自洽"
        # log-SNR 与旋转角的闭式关系：θ = atan(e^{-λ/2})
        assert abs(th - math.atan(math.exp(-l / 2.0))) < 1e-5, "θ 与 λ 的闭式关系不成立"

    # 均匀 β 有闭式：ᾱ'_t = ((1-β)^{t+1} - (1-β)^{n}) / (1 - (1-β)^{n})
    beta, n = 0.1, 6
    ab_u, _, _ = solve(torch.full((n,), beta))
    tail = (1.0 - beta) ** n
    for t, got in enumerate(ab_u.tolist()):
        want = ((1.0 - beta) ** (t + 1) - tail) / (1.0 - tail)
        assert abs(got - want) < 1e-6, f"第 {t} 项与均匀 β 闭式不符：{got} vs {want}"


def check_oracle_and_invalid(solve):
    gen = torch.Generator().manual_seed(2)
    for n, hi in ((3, 0.2), (7, 0.5), (40, 0.05)):
        betas = torch.rand(n, generator=gen, dtype=torch.float64) * hi
        got = solve(betas)
        want = _oracle(betas)
        names = ("alpha_bars", "log_snrs", "angles")
        for g, w, name in zip(got, want, names):
            assert torch.allclose(g, w.to(torch.float32), atol=1e-6, equal_nan=False), f"{name} 与 oracle 不一致"

    for bad in (
        lambda: solve([0.1, 0.2]),
        lambda: solve(torch.tensor([[0.1, 0.2]])),
        lambda: solve(torch.tensor([])),
        lambda: solve(torch.tensor([-0.1, 0.2])),
        lambda: solve(torch.tensor([0.1, 1.0])),
        lambda: solve(torch.tensor([0.1, 1.5])),
        lambda: solve(torch.tensor([0.1, float("nan")])),
        lambda: solve(torch.zeros(4)),
    ):
        try:
            bad()
            raise AssertionError("非法输入应抛 ValueError")
        except ValueError:
            pass
