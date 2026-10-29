import torch

LOGW = torch.tensor([-1.0, -2.0], dtype=torch.float64)
LOGL = torch.tensor([-3.0, -2.6], dtype=torch.float64)
REFW = torch.tensor([-1.2, -1.8], dtype=torch.float64)
REFL = torch.tensor([-2.8, -2.4], dtype=torch.float64)
BETA = 0.5
EXAMPLE_LOSS = 0.6799999999999997


def _oracle(lw, ll, rw, rl, beta):
    total = 0.0
    for i in range(len(lw)):
        d = (lw[i] - rw[i]) - (ll[i] - rl[i])
        total += (d - 1.0 / (2.0 * beta)) ** 2
    return total / len(lw)


def check_example(solve):
    y = solve(LOGW, LOGL, REFW, REFL, BETA)
    assert y.dim() == 0 and y.dtype == torch.float64, (y.dim(), y.dtype)
    assert abs(y.item() - EXAMPLE_LOSS) < 1e-12, y.item()


check_example.example = True


def check_oracle(solve):
    torch.manual_seed(13)
    for _ in range(8):
        n = 5
        lw = torch.randn(n, dtype=torch.float64).tolist()
        ll = torch.randn(n, dtype=torch.float64).tolist()
        rw = torch.randn(n, dtype=torch.float64).tolist()
        rl = torch.randn(n, dtype=torch.float64).tolist()
        beta = 0.3
        got = solve(torch.tensor(lw, dtype=torch.float64), torch.tensor(ll, dtype=torch.float64),
                    torch.tensor(rw, dtype=torch.float64), torch.tensor(rl, dtype=torch.float64), beta).item()
        exp = _oracle(lw, ll, rw, rl, beta)
        assert abs(got - exp) < 1e-12, (got, exp)


def check_zero_at_target(solve):
    # Δ 恰等于 1/(2β) 时损失为 0：β=0.5 -> 目标 1.0
    y = solve(torch.tensor([0.0], dtype=torch.float64), torch.tensor([-1.0], dtype=torch.float64),
              torch.tensor([0.0], dtype=torch.float64), torch.tensor([0.0], dtype=torch.float64), 0.5)
    assert y.item() == 0.0, y.item()
    # β=0.25 -> 目标 2.0
    y2 = solve(torch.tensor([0.0], dtype=torch.float64), torch.tensor([-2.0], dtype=torch.float64),
               torch.tensor([0.0], dtype=torch.float64), torch.tensor([0.0], dtype=torch.float64), 0.25)
    assert y2.item() == 0.0, y2.item()


def check_no_beta_scaling(solve):
    # IPO 不把损失整体乘 β：Δ=0 时损失 = (1/(2β))² 而不是 β·something
    y = solve(torch.tensor([0.0], dtype=torch.float64), torch.tensor([0.0], dtype=torch.float64),
              torch.tensor([0.0], dtype=torch.float64), torch.tensor([0.0], dtype=torch.float64), 0.5)
    assert abs(y.item() - 1.0) < 1e-12, y.item()
    y2 = solve(torch.tensor([0.0], dtype=torch.float64), torch.tensor([0.0], dtype=torch.float64),
               torch.tensor([0.0], dtype=torch.float64), torch.tensor([0.0], dtype=torch.float64), 0.25)
    assert abs(y2.item() - 4.0) < 1e-12, y2.item()


def check_ref_only_through_delta(solve):
    # 参考值整平移：两侧 ref 同时加 c，Δ 不变、损失不变
    base = solve(LOGW, LOGL, REFW, REFL, BETA).item()
    shifted = solve(LOGW, LOGL, REFW + 3.0, REFL + 3.0, BETA).item()
    assert abs(base - shifted) < 1e-12, (base, shifted)


def check_swap_invariance(solve):
    # 偏好侧与被拒侧整体对调（policy 与 ref 一起换），Δ 逐样本不变
    lw = torch.tensor([-1.0, -2.5], dtype=torch.float64)
    ll = torch.tensor([-2.0, -1.0], dtype=torch.float64)
    rw = torch.tensor([-1.5, -2.0], dtype=torch.float64)
    rl = torch.tensor([-1.5, -1.5], dtype=torch.float64)
    a = solve(lw, ll, rw, rl, 0.8).item()
    b = solve(ll, lw, rl, rw, 0.8).item()
    assert abs(a - b) < 1e-12, (a, b)


def check_float32_input(solve):
    y = solve(LOGW.to(torch.float32), LOGL.to(torch.float32),
              REFW.to(torch.float32), REFL.to(torch.float32), BETA)
    assert y.dtype == torch.float64
    assert abs(y.item() - EXAMPLE_LOSS) < 1e-6, y.item()


def check_errors(solve):
    good = torch.tensor([-1.0], dtype=torch.float64)

    def bad(fn, desc):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError(f"应抛 ValueError: {desc}")

    bad(lambda: solve([-1.0], good, good, good, 0.5), "非张量")
    bad(lambda: solve(good.unsqueeze(0), good, good, good, 0.5), "非一维")
    bad(lambda: solve(good, torch.tensor([-2.0, -2.0], dtype=torch.float64), good, good, 0.5), "形状不一致")
    bad(lambda: solve(good, good.to(torch.float32), good, good, 0.5), "dtype 不一致")
    bad(lambda: solve(torch.zeros(0), torch.zeros(0), torch.zeros(0), torch.zeros(0), 0.5), "空样本")
    bad(lambda: solve(good, good, good, good, 0.0), "beta=0")
    bad(lambda: solve(good, good, good, good, -1.0), "beta<0")
    bad(lambda: solve(torch.tensor([float("inf")]), good, good, good, 0.5), "含 Inf")
    bad(lambda: solve(good.to(torch.int64), good.to(torch.int64),
                      good.to(torch.int64), good.to(torch.int64), 0.5), "非浮点")
