import math
import torch


def _ls(x):
    return min(0.0, x) - math.log1p(math.exp(-abs(x)))


def _oracle(cw, crw, cl, crl, beta):
    """独立纯 python：手算隐式奖励差 + 稳定 logsigmoid + 均值。"""
    n = len(cw)
    tot = 0.0
    for i in range(n):
        logit = beta * ((cw[i] - crw[i]) - (cl[i] - crl[i]))
        tot += -_ls(logit)
    return tot / n


def check_example(solve):
    out = solve(
        torch.tensor([-1.2, -0.8, -2.0]),
        torch.tensor([-1.0, -1.0, -1.5]),
        torch.tensor([-2.5, -1.9, -1.0]),
        torch.tensor([-1.4, -1.6, -1.2]),
        0.1,
    )
    assert abs(out.item() - 0.682126) < 1e-5, out.item()


def check_symmetric_beta0(solve):
    # beta=0 → logits 全 0 → -logsigmoid(0)=ln2
    out = solve(torch.tensor([-1.0]), torch.tensor([0.0]), torch.tensor([-1.0]), torch.tensor([0.0]), 0.0)
    assert abs(out.item() - math.log(2)) < 1e-6, out.item()


def check_matches_oracle(solve):
    torch.manual_seed(1)
    for _ in range(50):
        N = int(torch.randint(1, 6, (1,)))
        cw = torch.randn(N)
        crw = torch.randn(N)
        cl = torch.randn(N)
        crl = torch.randn(N)
        beta = float(torch.rand(1)) * 0.5
        got = float(solve(cw, crw, cl, crl, beta).item())
        exp = _oracle(cw.tolist(), crw.tolist(), cl.tolist(), crl.tolist(), beta)
        assert abs(got - exp) < 1e-5, (got, exp)


def check_large_logit_stable(solve):
    # 极端 logit（chosen 远优于 ref）应稳定不 NaN
    out = solve(torch.tensor([100.0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]), 1.0)
    assert math.isfinite(out.item()) and out.item() < 1e-6, out.item()


check_example.example = True
