"""CorDA test cases. Judging convention: string CHECKS + check(solve)."""
import math

import torch

torch.set_default_dtype(torch.float64)


def check_damped_inv(solve):
    torch.manual_seed(0)
    # well-conditioned covariance resolves at the first damp step
    X = torch.randn(64, 5)
    cov = X.T @ X / 64
    fix, cov_inv, damp = solve(mode="damped_inv", cov=cov)
    assert damp == 0.01
    eye = torch.eye(5, dtype=torch.float64)
    assert torch.dist(fix @ cov_inv, eye) < 0.05
    assert torch.dist(cov_inv @ fix, eye) < 0.05
    # even a rank-1 covariance resolves at the first damp step (absolute-error tol)
    v = torch.randn(5, 1)
    cov1 = (v @ v.T)
    fix1, cov_inv1, damp1 = solve(mode="damped_inv", cov=cov1)
    assert damp1 == 0.01
    assert torch.dist(fix1 @ cov_inv1, torch.eye(5, dtype=torch.float64)) < 0.05
    # the starting damp is a parameter: starting at 0.02 returns 0.02 (still one step)
    _, _, damp2 = solve(mode="damped_inv", cov=cov1, damp=0.02)
    assert damp2 == 0.02
    # deterministic: same input replays the same damp
    _, _, damp1b = solve(mode="damped_inv", cov=cov1)
    assert damp1b == damp1
    # damp=0.01 base: compensate uses mean(diag(cov)) * damp
    torch.manual_seed(1)
    cov2 = torch.randn(4, 4).abs() / 4
    fix2, _, _ = solve(mode="damped_inv", cov=cov2)
    want = cov2 + torch.eye(4, dtype=torch.float64) * cov2.diagonal().mean() * 0.01
    assert torch.allclose(fix2, want, atol=1e-14)


def check_decompose_identity(solve):
    torch.manual_seed(2)
    m, n, r = 6, 5, 2
    W = torch.randn(m, n)
    X = torch.randn(64, n)
    cov = X.T @ X / 64
    # context identity holds for the FULL decomposition (r == rank): the covariance
    # cancels exactly, U diag(S) V^T == W. For r < rank it is a rank-r approximation
    # and the remainder lives in the frozen residual.
    U_f, S_f, V_f, resid_f, fix, cov_inv, damp = solve(
        mode="decompose", weight=W, cov=cov, r=min(m, n))
    recon = U_f @ torch.diag(S_f) @ V_f.t()
    assert (recon - W).norm() / W.norm() < 1e-5  # float32 svd precision
    assert torch.allclose(resid_f, torch.zeros_like(W.float()), atol=1e-5)

    # truncated r=2: value anchors (official float64 pipeline, seed 2 fixtures)
    U, S, V, resid, *_ = solve(mode="decompose", weight=W, cov=cov, r=r)
    assert U.shape == (m, r) and S.shape == (r,) and V.shape == (n, r)
    assert resid.shape == (m, n)
    assert abs(S[0].item() - 1.2007561922073364) < 1e-5, S[0].item()
    assert abs(U[0, 0].item() - 0.3145820200443268) < 1e-5, U[0, 0].item()
    assert abs(V[0, 0].item() - (-0.2790605425834656)) < 1e-5, V[0, 0].item()


def check_kpa_ipa(solve):
    torch.manual_seed(3)
    m, n, r = 6, 5, 2
    W = torch.randn(m, n)
    X = torch.randn(64, n)
    cov = X.T @ X / 64
    U_k, S_k, V_k, resid_k, *_ = solve(mode="decompose", weight=W, cov=cov, r=r,
                                       first_eigen=False)
    U_i, S_i, V_i, resid_i, *_ = solve(mode="decompose", weight=W, cov=cov, r=r,
                                       first_eigen=True)
    # KPA takes the last (smallest) r, IPA the first (largest) r — following PiSSA
    assert S_i[0] > S_k[-1], "IPA head singular value must dominate the KPA tail"
    assert not torch.equal(S_k, S_i)
    # both modes: adapter + frozen residual reconstruct W exactly
    for (Uu, Ss, Vv, rr) in [(U_k, S_k, V_k, resid_k), (U_i, S_i, V_i, resid_i)]:
        full = Uu @ torch.diag(Ss) @ Vv.t() + rr
        assert torch.allclose(full, W.float(), atol=1e-4)
    # frozen residual: KPA residual holds the dominant part (larger norm)
    assert resid_k.norm() > resid_i.norm()


def check_sigma_fuse(solve):
    torch.manual_seed(4)
    m, n, r = 5, 4, 2
    U = torch.randn(m, r)
    S = torch.rand(r) + 0.5
    V = torch.randn(n, r)
    want = U @ torch.diag(S) @ V.t()
    for fuse in ("UV", "U", "V"):
        a_w, b_w = solve(mode="fuse", u=U, s=S, v=V, fuse=fuse)
        assert a_w.shape == (m, r) and b_w.shape == (r, n), (fuse, a_w.shape, b_w.shape)
        assert torch.allclose(a_w @ b_w, want, atol=1e-10), fuse
    # UV mode: sqrt(S) split on both sides
    a_uv, b_uv = solve(mode="fuse", u=U, s=S, v=V, fuse="UV")
    assert torch.allclose(a_uv, U * S.sqrt().view(1, -1), atol=1e-14)
    assert torch.allclose(b_uv, V.t() * S.sqrt().view(-1, 1), atol=1e-14)
    # unknown fuse mode raises
    try:
        solve(mode="fuse", u=U, s=S, v=V, fuse="S")
        raise AssertionError("expected ValueError for unknown fuse mode")
    except ValueError:
        pass


def check_forward(solve):
    torch.manual_seed(5)
    m, n, r = 6, 5, 2
    W = torch.randn(m, n)
    X = torch.randn(64, n)
    cov = X.T @ X / 64
    U, S, V, resid, *_ = solve(mode="decompose", weight=W, cov=cov, r=r,
                               first_eigen=True)
    a_w, b_w = solve(mode="fuse", u=U, s=S, v=V, fuse="UV")
    x = torch.randn(4, n, dtype=torch.float32)
    y = solve(mode="forward", x=x, a_w=a_w, b_w=b_w, residual=resid)
    # manual three-term forward: BLinear -> ALinear + residual bypass
    y_manual = (x @ b_w.t()) @ a_w.t() + x @ resid.t()
    assert torch.equal(y, y_manual)
    # zero-delta property: at init the adapter output equals x @ W^T exactly
    assert torch.allclose(y, x @ W.float().t(), atol=1e-4)
    # KPA forward has the same zero-delta property
    U_k, S_k, V_k, resid_k, *_ = solve(mode="decompose", weight=W, cov=cov, r=r,
                                       first_eigen=False)
    a_k, b_k = solve(mode="fuse", u=U_k, s=S_k, v=V_k, fuse="UV")
    y_k = solve(mode="forward", x=x, a_w=a_k, b_w=b_k, residual=resid_k)
    assert torch.allclose(y_k, x @ W.float().t(), atol=1e-4)
    # error paths
    try:
        solve(mode="decompose", weight=W, cov=cov, r=0)
        raise AssertionError("expected ValueError for r=0")
    except ValueError:
        pass
    try:
        solve(mode="decompose", weight=W, cov=cov, r=7)
        raise AssertionError("expected ValueError for r>rank")
    except ValueError:
        pass
    # niter/seed sensitivity: same inputs replay bit-identically
    U2, S2, V2, *_ = solve(mode="decompose", weight=W, cov=cov, r=r, first_eigen=True)
    assert torch.equal(U2, U) and torch.equal(S2, S) and torch.equal(V2, V)


CHECKS = [
    "check_damped_inv",
    "check_decompose_identity",
    "check_kpa_ipa",
    "check_sigma_fuse",
    "check_forward",
]


def check(solve):
    for name in CHECKS:
        globals()[name](solve)
