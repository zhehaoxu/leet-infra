import math

import torch


def _oracle(P, Q):
    def f(A, B):
        tot = 0.0
        for i in range(A.shape[0]):
            rs = 0.0
            for c in range(A.shape[1]):
                a = float(A[i, c])
                b = float(B[i, c])
                if a > 0:
                    rs += a * math.log(a / b)
            tot += rs
        return tot / A.shape[0]
    return f(P, Q), f(Q, P)


def check_example(solve):
    P = torch.tensor([[0.7, 0.3], [0.2, 0.8]])
    Q = torch.tensor([[0.4, 0.6], [0.5, 0.5]])
    fwd, rev = solve(P, Q)
    assert abs(fwd.item() - 0.18826580) < 1e-5, f"fwd={fwd.item()}"
    assert abs(rev.item() - 0.20759282) < 1e-5, f"rev={rev.item()}"


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(3)
    P = torch.softmax(torch.randn(5, 4), dim=-1)
    Q = torch.softmax(torch.randn(5, 4), dim=-1)
    fwd, rev = solve(P, Q)
    of, orv = _oracle(P, Q)
    assert abs(fwd.item() - of) < 1e-5, (fwd.item(), of)
    assert abs(rev.item() - orv) < 1e-5, (rev.item(), orv)


def check_identical_zero(solve):
    P = torch.tensor([[0.5, 0.5], [0.2, 0.8]])
    fwd, rev = solve(P, P.clone())
    assert abs(fwd.item()) < 1e-6, fwd.item()
    assert abs(rev.item()) < 1e-6, rev.item()


def check_asymmetry(solve):
    torch.manual_seed(11)
    P = torch.softmax(torch.randn(6, 5), dim=-1)
    Q = torch.softmax(torch.randn(6, 5), dim=-1)
    fwd, rev = solve(P, Q)
    assert abs(fwd.item() - rev.item()) > 1e-4, "KL 非对称，前向不应等于反向"


def check_returns_scalar_pair(solve):
    P = torch.tensor([[0.6, 0.4]])
    Q = torch.tensor([[0.3, 0.7]])
    out = solve(P, Q)
    assert isinstance(out, tuple) and len(out) == 2, type(out)
    for x in out:
        assert torch.is_tensor(x) and x.dim() == 0, x.shape
