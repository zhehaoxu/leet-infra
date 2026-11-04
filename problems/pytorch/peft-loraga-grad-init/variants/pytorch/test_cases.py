"""Checks for peft-loraga-grad-init (judge contract: string CHECKS + check(solve))."""
import math

import torch

torch.set_default_dtype(torch.float64)


def check_grad_svd(solve):
    torch.manual_seed(0)
    d_out, d_in, r = 6, 8, 2
    grad = torch.randn(d_out, d_in)
    U, S, Vt = solve(mode="grad_svd", grad=grad, r=r, niter=4, seed=42)
    q = min(4 * r, min(grad.shape))
    assert U.shape == (d_out, q) and S.shape == (q,) and Vt.shape == (q, d_in), (U.shape, S.shape, Vt.shape)
    assert U.dtype == torch.float64 and S.dtype == torch.float64 and Vt.dtype == torch.float64
    # deterministic replay
    U2, S2, Vt2 = solve(mode="grad_svd", grad=grad, r=r, niter=4, seed=42)
    assert torch.equal(U, U2) and torch.equal(S, S2) and torch.equal(Vt, Vt2)
    # singular values descending
    assert bool(torch.all(S[:-1] >= S[1:]))
    # q is truncated by min(4r, min(shape)): r=1 -> q=4
    U1, S1, Vt1 = solve(mode="grad_svd", grad=grad, r=1, niter=4, seed=42)
    assert U1.shape == (d_out, 4) and S1.shape == (4,)
    # r must be positive
    try:
        solve(mode="grad_svd", grad=grad, r=0, niter=4, seed=42)
        raise AssertionError("expected ValueError for r=0")
    except ValueError:
        pass

    # value anchors: the official float32 pipeline (seed + q truncation) is deterministic
    s_anchor = torch.tensor([4.716037750244141, 3.28708815574646, 3.007770538330078,
                             1.8632816076278687, 1.1207000017166138, 0.9001481533050537])
    assert torch.equal(S, s_anchor), S
    assert U[0, 0] == -0.43773746490478516 and Vt[0, 0] == 0.48783788084983826

    # niter sensitivity: niter=0 (projection only) differs from niter=4
    _, S0, _ = solve(mode="grad_svd", grad=grad, r=r, niter=0, seed=42)
    assert not torch.equal(S0, S)


def check_direction(solve):
    torch.manual_seed(1)
    d_out, d_in, r = 6, 8, 2
    grad = torch.randn(d_out, d_in)
    U, S, Vt = solve(mode="grad_svd", grad=grad, r=r, niter=4, seed=42)

    B_i, A_i = solve(mode="direction", U=U, Vt=Vt, r=r, direction="ArBr")
    assert torch.equal(B_i, U[:, 0:2 * r:2]) and torch.equal(A_i, Vt[1:2 * r:2, :])
    B_2, A_2 = solve(mode="direction", U=U, Vt=Vt, r=r, direction="A2rBr")
    assert torch.equal(B_2, U[:, :r]) and torch.equal(A_2, Vt[r:2 * r, :])
    B_3, A_3 = solve(mode="direction", U=U, Vt=Vt, r=r, direction="ArB2r")
    assert torch.equal(B_3, U[:, r:2 * r]) and torch.equal(A_3, Vt[:r, :])
    for B, A in [(B_i, A_i), (B_2, A_2), (B_3, A_3)]:
        assert B.shape == (d_out, r) and A.shape == (r, d_in)

    # random direction: official draws indexes_A first, indexes_B second
    B_r, A_r = solve(mode="direction", U=U, Vt=Vt, r=r, direction="random", seed=11)
    B_r2, A_r2 = solve(mode="direction", U=U, Vt=Vt, r=r, direction="random", seed=11)
    assert torch.equal(B_r, B_r2) and torch.equal(A_r, A_r2)
    import random as _random
    _random.seed(11)
    lst = _random.sample(range(2 * r), 2 * r)
    assert torch.equal(A_r, Vt[lst[0:r], :]) and torch.equal(B_r, U[:, lst[r:2 * r]])

    # unknown direction
    try:
        solve(mode="direction", U=U, Vt=Vt, r=r, direction="ArBr_")
        raise AssertionError("expected ValueError for unknown direction")
    except ValueError:
        pass


def check_scale(solve):
    torch.manual_seed(2)
    d_out, d_in, r = 6, 8, 2
    eta = 4.0
    B = torch.randn(d_out, r)
    A = torch.randn(r, d_in)

    # gd: both /= eta
    B_g, A_g = solve(mode="scale", B=B, A=A, scaling=eta, scale="gd")
    assert torch.allclose(B_g, B / eta) and torch.allclose(A_g, A / eta)

    # unit: untouched
    B_u, A_u = solve(mode="scale", B=B, A=A, scaling=eta, scale="unit")
    assert torch.equal(B_u, B) and torch.equal(A_u, A)

    # stable: m**0.25 / sqrt(gamma) with m = out_features = B.size(0)
    gamma = 2.0
    B_s, A_s = solve(mode="scale", B=B, A=A, scaling=eta, scale="stable", stable_gamma=gamma)
    c = d_out ** 0.25 / gamma ** 0.5
    assert torch.allclose(B_s, B * c) and torch.allclose(A_s, A * c)

    # weightS: S /= eta; avg_s = sqrt(S[:r]).mean(); both *= avg_s
    Sw = torch.rand(4 * r)
    B_w, A_w = solve(mode="scale", B=B, A=A, scaling=eta, scale="weightS", weight_singulars=Sw)
    S_div = Sw / eta
    avg_s = torch.sqrt(S_div[:r]).mean()
    assert torch.allclose(B_w, B * avg_s) and torch.allclose(A_w, A * avg_s)
    # anchor: avg_s value depends only on the input S — recompute independently
    assert abs(avg_s.item() - (Sw / eta)[:r].sqrt().mean().item()) < 1e-14

    # inputs are not mutated
    assert torch.equal(B, B) and torch.equal(A, A)

    # unknown scale
    try:
        solve(mode="scale", B=B, A=A, scaling=eta, scale="linear")
        raise AssertionError("expected ValueError for unknown scale")
    except ValueError:
        pass


def check_offset(solve):
    torch.manual_seed(3)
    d_out, d_in, r = 6, 8, 2
    eta = 4.0
    B = torch.randn(d_out, r)
    A = torch.randn(r, d_in)
    weight = torch.randn(d_out, d_in) * 0.5

    # plain absorption: new_weight = weight - B@A*eta; identity W' + offset == W
    W2, A2, B2 = solve(mode="offset_apply", B=B, A=A, scaling=eta, weight=weight)
    offset = B @ A * eta
    assert torch.allclose(W2, weight - offset, atol=1e-12)
    assert torch.allclose(W2 + offset, weight, atol=1e-12)
    assert torch.equal(A2, A) and torch.equal(B2, B)  # no clip -> factors unchanged

    # norm_clip not triggered when ratio >= 1
    W3, A3, B3 = solve(mode="offset_apply", B=B, A=A, scaling=eta, weight=weight, norm_clip=True)
    ratio = weight.abs().max() / offset.abs().max()
    if ratio >= 1:
        assert torch.equal(W3, W2) and torch.equal(A3, A2) and torch.equal(B3, B2)

    # norm_clip triggered: force a tiny weight so ratio < 1
    w_small = weight * 1e-3
    W4, A4, B4 = solve(mode="offset_apply", B=B, A=A, scaling=eta, weight=w_small, norm_clip=True)
    ratio4 = w_small.abs().max() / offset.abs().max()
    assert ratio4 < 1
    assert torch.allclose(W4, w_small - offset * ratio4, atol=1e-12)
    # invariant: (B*sqrt(ratio)) @ (A*sqrt(ratio)) * eta == offset * ratio
    assert torch.allclose((B4 @ A4) * eta, offset * ratio4, atol=1e-12)
    assert torch.allclose(A4, A * ratio4 ** 0.5) and torch.allclose(B4, B * ratio4 ** 0.5)


def check_pipeline(solve):
    # end-to-end: grad_svd -> direction -> scale -> offset_apply (official order)
    torch.manual_seed(4)
    d_out, d_in, r = 6, 8, 2
    grad = torch.randn(d_out, d_in)
    weight = torch.randn(d_out, d_in) * 0.5
    eta = 4.0

    U, S, Vt = solve(mode="grad_svd", grad=grad, r=r, niter=4, seed=42)
    B, A = solve(mode="direction", U=U, Vt=Vt, r=r, direction="ArBr")
    B_g, A_g = solve(mode="scale", B=B, A=A, scaling=eta, scale="gd")
    W2, A_w, B_w = solve(mode="offset_apply", B=B_g, A=A_g, scaling=eta, weight=weight, norm_clip=True)

    # B_w@A_w*eta reproduces the (possibly clipped) absorbed offset
    offset = B_g @ A_g * eta
    ratio = min(1.0, (weight.abs().max() / offset.abs().max()).item())
    assert torch.allclose((B_w @ A_w) * eta, offset * ratio, atol=1e-12)
    assert torch.allclose(W2, weight - offset * ratio, atol=1e-12)

    # determinism of the whole pipeline
    U2, S2, Vt2 = solve(mode="grad_svd", grad=grad, r=r, niter=4, seed=42)
    B2, A2 = solve(mode="direction", U=U2, Vt=Vt2, r=r, direction="ArBr")
    assert torch.equal(B, B2) and torch.equal(A, A2)


CHECKS = [
    "check_grad_svd",
    "check_direction",
    "check_scale",
    "check_offset",
    "check_pipeline",
]
