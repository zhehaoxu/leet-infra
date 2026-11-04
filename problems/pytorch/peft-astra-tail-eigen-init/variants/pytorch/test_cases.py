"""Astra test cases. Judging convention: string CHECKS + check(solve)."""
import math

import torch

torch.set_default_dtype(torch.float64)


def check_cov_accumulate(solve):
    torch.manual_seed(0)
    # per-call global-max normalization + accumulation + sample mean
    b1 = torch.randn(16, 6)
    b2 = torch.randn(16, 6) * 5.0
    cov, cnt = solve(mode="cov_accumulate", batches=[b1, b2])
    assert cnt == 2
    assert cov.dtype == torch.float32, "official pipeline forces the float32 cast"
    # official quirk: torch.max(out).abs() takes the SIGNED max first — the divisor is
    # |max(out)|, NOT out.abs().max() (a mutation magnet, and different when negatives dominate)
    n1 = b1 / b1.max().abs()
    n2 = b2 / b2.max().abs()
    want = (n1.t() @ n1 + n2.t() @ n2) / 2
    assert torch.allclose(cov.float(), want.float(), atol=1e-5), "hook account mismatch"
    # normalization makes the covariance invariant to the batch magnitude
    cov_big, _ = solve(mode="cov_accumulate", batches=[b1, b2 * 100])
    assert torch.allclose(cov_big, cov, atol=1e-5), "magnitude invariance broken"
    # a single call reproduces the plain normalized second moment
    cov1, cnt1 = solve(mode="cov_accumulate", batches=[b1])
    assert cnt1 == 1
    assert torch.allclose(cov1.double(), n1.t() @ n1, atol=1e-5)
    # signed-max vs abs-max: an all-negative batch exposes the difference
    bneg = -torch.abs(torch.randn(16, 6)) - 0.5
    cov_n, _ = solve(mode="cov_accumulate", batches=[bneg])
    n_neg = bneg / bneg.max().abs()   # |max| is the smallest |entry| band here
    assert torch.allclose(cov_n.double(), n_neg.t() @ n_neg, atol=1e-5)
    assert (bneg.abs().max() > bneg.max().abs())  # the two divisors genuinely differ
    # NaN in activations raises
    bad = b1.clone()
    bad[0, 0] = float("nan")
    try:
        solve(mode="cov_accumulate", batches=[bad])
        raise AssertionError("expected ValueError for NaN activations")
    except ValueError:
        pass
    # 3-D input (batch, seq, dim) is flattened like the official reshape
    b3 = torch.randn(2, 8, 6)
    cov3, _ = solve(mode="cov_accumulate", batches=[b3])
    f3 = b3.reshape(-1, 6) / b3.max().abs()
    want3 = f3.t() @ f3
    assert torch.allclose(cov3.double(), want3, atol=1e-5)


def check_eigens(solve):
    torch.manual_seed(2)
    torch.randn(6, 5, dtype=torch.float32)  # canonical RNG order: W first (see check_init)
    X = torch.randn(32, 6, dtype=torch.float32)
    cov, _ = solve(mode="cov_accumulate", batches=[X])
    S, V_r = solve(mode="eigens", cov=cov, r=2, out_dim=6)
    # eigh is ascending: S[-1] is the largest eigenvalue, S kept FULL length
    assert S.size(0) == 6
    assert S[-1] >= S[0]
    # official float32 pipeline value anchors
    assert abs(S[0].item() - 1.806457757949829) < 1e-4, S[0].item()
    assert abs(S[-1].item() - 8.876955032348633) < 1e-3, S[-1].item()
    assert abs(V_r[0, 0].item() - 0.7613240480422974) < 1e-4, V_r[0, 0].item()
    # V_r spans the top-r eigenspace: rayleigh quotients match the top eigenvalues
    lam = V_r.t() @ cov.float() @ V_r
    assert torch.allclose(lam, torch.diag(S[-2:]), atol=1e-2)
    # orthogonality of the cropped basis
    assert torch.allclose(V_r.t() @ V_r, torch.eye(2, dtype=torch.float32), atol=1e-5)
    # error paths
    try:
        solve(mode="eigens", cov=cov, r=7, out_dim=6)
        raise AssertionError("expected ValueError for r>out_dim")
    except ValueError:
        pass
    try:
        solve(mode="eigens", cov=cov, r=0)
        raise AssertionError("expected ValueError for r=0")
    except ValueError:
        pass


def check_init(solve):
    torch.manual_seed(2)
    W = torch.randn(6, 5, dtype=torch.float32)
    X = torch.randn(32, 6, dtype=torch.float32)
    cov, _ = solve(mode="cov_accumulate", batches=[X])
    S, V_r = solve(mode="eigens", cov=cov, r=2, out_dim=6)
    lora_a, lora_b, w_new = solve(mode="init", weight=W, v_r=V_r, scaling=2.0)
    # official account: A = V^T W (r, in), B = V (out, r)
    assert lora_a.shape == (2, 5) and lora_b.shape == (6, 2)
    assert torch.allclose(lora_a, (V_r.t() @ W), atol=1e-5)
    assert torch.equal(lora_b, V_r)
    # value anchors (float32 pipeline)
    assert abs(lora_a[0, 0].item() - (-0.8271405696868896)) < 1e-5, lora_a[0, 0].item()
    assert abs(w_new[0, 0].item() - 0.5683993101119995) < 1e-5, w_new[0, 0].item()
    # residualization: W_new + scaling * B@A == W (float32 domain)
    assert torch.allclose(w_new + 2.0 * lora_b @ lora_a, W, atol=1e-5)
    # B@A is exactly the projection of W onto the V subspace
    assert torch.allclose(lora_b @ lora_a, V_r @ (V_r.t() @ W), atol=1e-5)
    # error paths
    try:
        solve(mode="init", weight=W, v_r=torch.randn(6, 7, dtype=torch.float32), scaling=1.0)
        raise AssertionError("expected ValueError for r>out_dim")
    except ValueError:
        pass
    try:
        solve(mode="init", weight=W, v_r=torch.randn(5, 2, dtype=torch.float32), scaling=1.0)
        raise AssertionError("expected ValueError for V shape mismatch")
    except ValueError:
        pass
    bad_v = V_r.clone()
    bad_v[0, 0] = float("inf")
    try:
        solve(mode="init", weight=W, v_r=bad_v, scaling=1.0)
        raise AssertionError("expected ValueError for Inf in V")
    except ValueError:
        pass
    # official dtype gate: float64 weights are rejected with TypeError
    try:
        solve(mode="init", weight=W.double(), v_r=V_r.double(), scaling=1.0)
        raise AssertionError("expected TypeError for float64 weight")
    except TypeError:
        pass


def check_zero_delta(solve):
    torch.manual_seed(2)
    out_dim, in_dim, r = 6, 5, 2
    W = torch.randn(out_dim, in_dim, dtype=torch.float32)
    X = torch.randn(32, out_dim, dtype=torch.float32)
    cov, _ = solve(mode="cov_accumulate", batches=[X])
    S, V_r = solve(mode="eigens", cov=cov, r=r, out_dim=out_dim)
    lora_a, lora_b, w_new = solve(mode="init", weight=W, v_r=V_r, scaling=3.0)
    x = torch.randn(4, in_dim, dtype=torch.float32)
    y = solve(mode="forward", x=x, lora_a=lora_a, lora_b=lora_b,
              w_new=w_new, scaling=3.0)
    # manual three-term forward
    y_manual = 3.0 * (x @ lora_a.t() @ lora_b.t()) + x @ w_new.t()
    assert torch.allclose(y, y_manual, atol=1e-5)
    # zero-delta property: the initialized model reproduces the original output
    assert torch.allclose(y, x @ W.float().t(), atol=1e-4)
    # same for scaling=1
    la1, lb1, wn1 = solve(mode="init", weight=W, v_r=V_r)
    y1 = solve(mode="forward", x=x, lora_a=la1, lora_b=lb1, w_new=wn1)
    assert torch.allclose(y1, x @ W.float().t(), atol=1e-4)


CHECKS = [
    "check_cov_accumulate",
    "check_eigens",
    "check_init",
    "check_zero_delta",
]


def check(solve):
    for name in CHECKS:
        globals()[name](solve)
