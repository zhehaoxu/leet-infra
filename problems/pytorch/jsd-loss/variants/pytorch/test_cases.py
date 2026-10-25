import math

import torch


def _kl_vec(p, m):
    rs = 0.0
    for pc, mc in zip(p, m):
        if pc > 0:
            rs += pc * math.log(pc / mc)
    return rs


def _oracle(P, Q, beta):
    rows = P.tolist()
    qrows = Q.tolist()
    tot = 0.0
    for prow, qrow in zip(rows, qrows):
        m = [beta * p + (1 - beta) * q for p, q in zip(prow, qrow)]
        tot += beta * _kl_vec(prow, m) + (1 - beta) * _kl_vec(qrow, m)
    return tot / len(rows)


def check_example_bound(solve):
    P = torch.tensor([[1.0, 0.0]])
    Q = torch.tensor([[0.0, 1.0]])
    out = solve(P, Q, 0.5)
    assert abs(out.item() - math.log(2)) < 1e-5, out.item()


check_example_bound.example = True


def check_example_pair(solve):
    P = torch.tensor([[0.7, 0.3], [0.2, 0.8]])
    Q = torch.tensor([[0.4, 0.6], [0.5, 0.5]])
    out = solve(P, Q, 0.5)
    assert abs(out.item() - 0.04843632) < 1e-5, out.item()


def check_identical_zero(solve):
    P = torch.tensor([[0.5, 0.5], [0.2, 0.8]])
    out = solve(P, P.clone(), 0.5)
    assert abs(out.item()) < 1e-6, out.item()


def check_symmetry(solve):
    torch.manual_seed(7)
    P = torch.softmax(torch.randn(4, 5), dim=-1)
    Q = torch.softmax(torch.randn(4, 5), dim=-1)
    assert abs(solve(P, Q, 0.5).item() - solve(Q, P, 0.5).item()) < 1e-6


def check_matches_oracle(solve):
    torch.manual_seed(9)
    P = torch.softmax(torch.randn(6, 4), dim=-1)
    Q = torch.softmax(torch.randn(6, 4), dim=-1)
    for beta in (0.5, 0.3, 0.7):
        assert abs(solve(P, Q, beta).item() - _oracle(P, Q, beta)) < 1e-5, beta
