"""PSOFT test cases — judge-style: check(solve) + string CHECKS."""
import math

import torch

torch.set_default_dtype(torch.float64)


def check_skew_symmetric(solve):
    size = 4
    n = size * (size - 1) // 2
    w = torch.randn(n)
    Q = solve(mode="skew_symmetric", weight=w, size=size)
    assert Q.shape == (size, size), Q.shape
    # skew-symmetric: Q^T == -Q, zero diagonal
    assert torch.equal(Q, -Q.T)
    assert torch.count_nonzero(torch.diagonal(Q)).item() == 0
    # upper triangle holds the parameters in order
    rows, cols = torch.triu_indices(size, size, 1)
    for k in range(n):
        assert Q[rows[k], cols[k]].item() == w[k].item()
        assert Q[cols[k], rows[k]].item() == -w[k].item()


def check_orth_matrix(solve):
    size = 4
    n = size * (size - 1) // 2
    torch.manual_seed(0)
    w = torch.randn(n) * 0.1
    # Cayley solve branch: R = (I+Q)(I-Q)^-1, orthogonal
    R = solve(mode="orth_matrix", weight=w, size=size)
    assert R.shape == (size, size)
    assert torch.allclose(R @ R.T, torch.eye(size), atol=1e-9), (R @ R.T)
    assert torch.allclose(R.T @ R, torch.eye(size), atol=1e-9)
    # det = +1 (rotation, connected to identity)
    assert abs(torch.det(R).item() - 1.0) < 1e-9
    # zero params => R = I
    R0 = solve(mode="orth_matrix", weight=torch.zeros(n), size=size)
    assert torch.allclose(R0, torch.eye(size), atol=1e-12)
    # manual: solve(X, I-Q) = I+Q  <=>  X = (I+Q) @ inv(I-Q)
    Q = solve(mode="skew_symmetric", weight=w, size=size)
    eye = torch.eye(size)
    want = (eye + Q) @ torch.inverse(eye - Q)
    assert torch.allclose(R, want, atol=1e-9, rtol=0)
    # anchor
    assert abs(R[0, 0].item() - 0.8674077806164663) < 1e-9, R[0, 0].item()

    # Neumann branch: R = I + 2Q + 2Q^2 + 2Q^3 + ... + Q^(t-1)
    t = 5
    Rn = solve(mode="orth_matrix", weight=w, size=size, use_cayley_neumann=True,
               num_cayley_neumann_terms=t)
    Qp = eye.clone() + Q * 2.0
    Q2 = Q @ Q
    Qp.add_(Q2, alpha=2.0)
    Q3 = Q2 @ Q
    Qp.add_(Q3, alpha=2.0)
    Qp.add_(Q3 @ Q)  # final term coefficient 1.0
    assert torch.allclose(Rn, Qp, atol=1e-12, rtol=0)
    # not orthogonal in general (truncated series)
    assert not torch.allclose(Rn @ Rn.T, eye, atol=1e-6)
    # projection: Q rescaled when its Frobenius norm exceeds eps
    Rp = solve(mode="orth_matrix", weight=w * 100, size=size, use_cayley_neumann=True,
               num_cayley_neumann_terms=t, cayley_neumann_eps=0.9)
    Qbig = Q * 100.0
    Qp2 = Qbig * (0.9 / (torch.linalg.norm(Qbig, ord="fro") + 1e-12))
    want_p = eye.clone() + Qp2 * 2.0
    Qp2_2 = Qp2 @ Qp2
    want_p.add_(Qp2_2, alpha=2.0)
    want_p.add_(Qp2_2 @ Qp2, alpha=2.0)
    want_p.add_(Qp2_2 @ Qp2 @ Qp2)
    assert torch.allclose(Rp, want_p, atol=1e-12, rtol=0)
    # no projection without eps even for large params
    Rbig = solve(mode="orth_matrix", weight=w * 100, size=size, use_cayley_neumann=True,
                 num_cayley_neumann_terms=t)
    Qb = solve(mode="skew_symmetric", weight=w * 100, size=size)
    want_b = eye.clone() + Qb * 2.0
    Qb2 = Qb @ Qb
    want_b.add_(Qb2, alpha=2.0)
    Qb3 = Qb2 @ Qb
    want_b.add_(Qb3, alpha=2.0)
    want_b.add_(Qb3 @ Qb)
    assert torch.allclose(Rbig, want_b, atol=1e-9, rtol=0)


def check_magnify(solve):
    torch.manual_seed(1)
    size = 3
    R = torch.randn(size, size)
    vb = torch.randn(size)
    va = torch.randn(size)
    out = solve(mode="magnify", R=R, vector_b=vb, vector_a=va)
    assert torch.equal(out, vb[:, None] * R * va[None, :])
    # row vs column scaling direction
    only_b = solve(mode="magnify", R=R, vector_b=vb, vector_a=torch.ones(size))
    assert torch.equal(only_b, vb[:, None] * R)
    only_a = solve(mode="magnify", R=R, vector_b=torch.ones(size), vector_a=va)
    assert torch.equal(only_a, R * va[None, :])
    # ones vectors are identity
    assert torch.equal(solve(mode="magnify", R=R, vector_b=torch.ones(size),
                             vector_a=torch.ones(size)), R)


def check_svd_init(solve):
    torch.manual_seed(2)
    out_f, in_f, r = 6, 8, 3
    W = torch.randn(out_f, in_f)
    f = solve(mode="svd_factors", W=W, r=r)
    Vr, Sr, Uhr = f["Vr"], f["Sr"], f["Uhr"]
    assert Vr.shape == (out_f, r) and Sr.shape == (r,) and Uhr.shape == (r, in_f)
    # truncated SVD reconstructs the best rank-r approximation
    approx = Vr @ torch.diag(Sr) @ Uhr
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    assert torch.allclose(approx, U[:, :r] @ torch.diag(S[:r]) @ Vh[:r], atol=1e-9)
    # scaling moves into B for psoft_init (A keeps raw Uhr)
    scaling = 2.0
    ab = solve(mode="init_ab", factors=f, scaling=scaling, init_type="psoft_init")
    A, B = ab["A"], ab["B"]
    assert torch.equal(A, Uhr)
    assert torch.allclose(B, Vr @ torch.diag(Sr / scaling), atol=1e-12, rtol=0)
    # pissa_init splits singular values by sqrt
    ab2 = solve(mode="init_ab", factors=f, scaling=scaling, init_type="pissa_init")
    s = torch.sqrt(Sr / scaling)
    assert torch.allclose(ab2["A"], torch.diag(s) @ Uhr, atol=1e-12, rtol=0)
    assert torch.allclose(ab2["B"], Vr @ torch.diag(s), atol=1e-12, rtol=0)
    # both inits give the same full-rank product B @ A (== truncated W / scaling)
    assert torch.allclose(B @ A, ab2["B"] @ ab2["A"], atol=1e-9)
    # unknown init raises
    try:
        solve(mode="init_ab", factors=f, scaling=scaling, init_type="nope")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    # anchors
    assert abs(Vr[0, 0].item() - 0.36490302916512796) < 1e-9, Vr[0, 0].item()
    assert abs(Sr[0].item() - 4.971680643952505) < 1e-9, Sr[0].item()


def check_delta_branch(solve):
    torch.manual_seed(3)
    out_f, in_f, r = 6, 8, 3
    scaling = 0.5
    W = torch.randn(out_f, in_f)
    f = solve(mode="svd_factors", W=W, r=r)
    ab = solve(mode="init_ab", factors=f, scaling=scaling, init_type="psoft_init")
    A, B = ab["A"], ab["B"]
    n = r * (r - 1) // 2
    w = torch.randn(n) * 0.1
    R = solve(mode="orth_matrix", weight=w, size=r)
    dW = solve(mode="delta", B=B, R=R, A=A, r=r, scaling=scaling)
    assert dW.shape == (out_f, in_f), dW.shape
    want = (B @ (R - torch.eye(r)) @ A) * scaling
    assert torch.allclose(dW, want, atol=1e-12, rtol=0)
    # R = I => zero delta (identity at init)
    dW0 = solve(mode="delta", B=B, R=torch.eye(r), A=A, r=r, scaling=scaling)
    assert torch.count_nonzero(dW0).item() == 0
    # branch consistency: (x @ dW.T) == branch(x) for the same R
    x = torch.randn(2, 5, in_f)
    br = solve(mode="branch", x=x, A=A, B=B, R=R, scaling=scaling)
    assert br.shape == (2, 5, out_f), br.shape
    want_br = (x @ dW.T)
    assert torch.allclose(br, want_br, atol=1e-9, rtol=0)
    # manual branch: xa = x A^T; xr = xa R^T; (xr - xa) B^T * scaling
    xa = x @ A.t()
    xr = xa @ R.t()
    assert torch.allclose(br, (xr - xa) @ B.t() * scaling, atol=1e-12, rtol=0)
    # R = I => zero branch
    br0 = solve(mode="branch", x=x, A=A, B=B, R=torch.eye(r), scaling=scaling)
    assert torch.count_nonzero(br0).item() == 0
    # anchors
    assert abs(dW[0, 0].item() - 0.016577474832680194) < 1e-9, dW[0, 0].item()
    assert abs(br[0, 0, 0].item() - 0.3148390436622794) < 1e-9, br[0, 0, 0].item()


def check_r_check(solve):
    assert solve(mode="r_check", r=4, in_features=8, out_features=6) is True
    assert solve(mode="r_check", r=6, in_features=8, out_features=6) is True
    try:
        solve(mode="r_check", r=7, in_features=8, out_features=6)
        raise AssertionError("expected ValueError for r > min(in, out)")
    except ValueError:
        pass


CHECKS = [
    "check_skew_symmetric",
    "check_orth_matrix",
    "check_magnify",
    "check_svd_init",
    "check_delta_branch",
    "check_r_check",
]


def check(solve):
    for name in CHECKS:
        globals()[name](solve)
