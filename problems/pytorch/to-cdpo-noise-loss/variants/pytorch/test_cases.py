import math

import torch

LOGW = torch.tensor([-1.0], dtype=torch.float64)
LOGL = torch.tensor([-3.0], dtype=torch.float64)
REFW = torch.tensor([-1.2], dtype=torch.float64)
REFL = torch.tensor([-2.8], dtype=torch.float64)
BETA = 0.5
EPS = 0.1
EXAMPLE_LOSS = 0.6181388693815918
EXAMPLE_DPO = 0.5981388693815918


def _oracle(lw, ll, rw, rl, beta, eps):
    total = 0.0
    for i in range(len(lw)):
        x = beta * ((lw[i] - rw[i]) - (ll[i] - rl[i]))
        total += -(1.0 - eps) * (-math.log1p(math.exp(-x))) - eps * (-math.log1p(math.exp(x)))
    return total / len(lw)


def check_example(solve):
    y = solve(LOGW, LOGL, REFW, REFL, BETA, EPS)
    assert y.dim() == 0 and y.dtype == torch.float64, (y.dim(), y.dtype)
    assert abs(y.item() - EXAMPLE_LOSS) < 1e-12, y.item()


check_example.example = True


def check_eps_zero_is_dpo(solve):
    y = solve(LOGW, LOGL, REFW, REFL, BETA, 0.0)
    assert abs(y.item() - EXAMPLE_DPO) < 1e-12, y.item()


def check_zero_logit_ln2(solve):
    # Δ=0 -> logit=0，任意 eps 损失都是 ln2
    for eps in (0.0, 0.1, 0.25, 0.5):
        y = solve(torch.tensor([-1.0], dtype=torch.float64), torch.tensor([-2.0], dtype=torch.float64),
                  torch.tensor([-1.0], dtype=torch.float64), torch.tensor([-2.0], dtype=torch.float64), 1.0, eps)
        assert abs(y.item() - math.log(2.0)) < 1e-12, (eps, y.item())


def check_oracle(solve):
    torch.manual_seed(17)
    for _ in range(8):
        n = 5
        lw = torch.randn(n, dtype=torch.float64).tolist()
        ll = torch.randn(n, dtype=torch.float64).tolist()
        rw = torch.randn(n, dtype=torch.float64).tolist()
        rl = torch.randn(n, dtype=torch.float64).tolist()
        for eps in (0.05, 0.2, 0.5):
            got = solve(torch.tensor(lw, dtype=torch.float64),
                        torch.tensor(ll, dtype=torch.float64),
                        torch.tensor(rw, dtype=torch.float64),
                        torch.tensor(rl, dtype=torch.float64), 0.7, eps).item()
            exp = _oracle(lw, ll, rw, rl, 0.7, eps)
            assert abs(got - exp) < 1e-12, (eps, got, exp)


def check_noise_increases_loss(solve):
    # logit>0（已分对）时，标签噪声一定抬高损失
    x = solve(LOGW, LOGL, REFW, REFL, BETA, 0.0).item()
    for eps in (0.05, 0.2, 0.5):
        y = solve(LOGW, LOGL, REFW, REFL, BETA, eps).item()
        assert y > x, (eps, y, x)


def check_symmetric_eps_half(solve):
    # eps=0.5 时软标签对称，损失 = -0.5(logσ(x)+logσ(-x))
    x = 0.7
    y = solve(torch.tensor([0.0], dtype=torch.float64), torch.tensor([-x], dtype=torch.float64),
              torch.tensor([0.0], dtype=torch.float64),
              torch.tensor([0.0], dtype=torch.float64), 1.0, 0.5).item()
    exp = -0.5 * (-math.log1p(math.exp(-x)) - math.log1p(math.exp(x)))
    assert abs(y - exp) < 1e-12, (y, exp)


def check_float32_input(solve):
    y = solve(LOGW.to(torch.float32), LOGL.to(torch.float32),
              REFW.to(torch.float32), REFL.to(torch.float32), BETA, EPS)
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

    bad(lambda: solve(good, good, good, good, 0.5, -0.1), "eps 为负")
    bad(lambda: solve(good, good, good, good, 0.5, 0.51), "eps 超过 0.5")
    bad(lambda: solve(good, good, good, good, 0.0, 0.1), "beta=0")
    bad(lambda: solve([-1.0], good, good, good, 0.5, 0.1), "非张量")
    bad(lambda: solve(good.unsqueeze(0), good, good, good, 0.5, 0.1), "非一维")
    bad(lambda: solve(good, torch.tensor([-2.0, -2.0]), good, good, 0.5, 0.1), "形状不一致")
    bad(lambda: solve(torch.zeros(0), torch.zeros(0), torch.zeros(0), torch.zeros(0), 0.5, 0.1), "空样本")
    bad(lambda: solve(torch.tensor([float("nan")]), good, good, good, 0.5, 0.1), "含 NaN")
