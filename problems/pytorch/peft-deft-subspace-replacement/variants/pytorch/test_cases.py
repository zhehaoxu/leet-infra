"""DeFT test cases — judge-style: check(solve) + string CHECKS."""
import math

import torch

torch.set_default_dtype(torch.float64)


def check_resolve(solve):
    assert solve(mode="resolve_r", r=8, out_features=64) == 8
    # effective rank capped at out_features
    assert solve(mode="resolve_r", r=8, out_features=2) == 2
    try:
        solve(mode="resolve_r", r=0, out_features=8)
        raise AssertionError("expected ValueError for r <= 0")
    except ValueError:
        pass
    # scaling = alpha / r; 1.0 without alpha
    assert solve(mode="resolve_scaling", alpha=4.0, r=8) == 0.5
    assert solve(mode="resolve_scaling", r=8) == 1.0
    assert solve(mode="resolve_scaling", alpha=None, r=8) == 1.0


def check_project(solve):
    torch.manual_seed(0)
    out_f, r = 6, 3
    P = torch.randn(out_f, r)
    # qr branch: orthonormal Q_P, right == Q_P, orthogonal projector
    out = solve(mode="project", P=P, method="qr")
    Q_P, right = out["Q_P"], out["right"]
    assert Q_P.shape == (out_f, r) and right.shape == (out_f, r)
    assert torch.allclose(Q_P.T @ Q_P, torch.eye(r, dtype=torch.float32), atol=1e-5, rtol=0)
    assert torch.equal(right, Q_P)
    P_proj = Q_P @ right.T
    assert torch.allclose(P_proj @ P_proj, P_proj, atol=1e-6, rtol=0)  # idempotent (fp32) (fp32)
    # relu branch: non-orthogonal, right = relu(P)
    out2 = solve(mode="project", P=P, method="relu")
    assert torch.equal(out2["Q_P"], P.to(torch.float32))
    assert torch.equal(out2["right"], torch.relu(P.to(torch.float32)))
    assert not torch.allclose(out2["Q_P"].T @ out2["Q_P"], torch.eye(r, dtype=torch.float32), atol=1e-3)
    # unknown method raises
    try:
        solve(mode="project", P=P, method="nope")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    # anchor (qr)
    assert abs(Q_P[0, 0].item() - (-0.8325183391571045)) < 1e-9, Q_P[0, 0].item()


def check_delta_identity(solve):
    torch.manual_seed(1)
    out_f, in_f, r = 6, 8, 3
    scaling = 0.5
    W = torch.randn(out_f, in_f)
    P = torch.randn(out_f, r) * 0.1
    for method in ("qr", "relu"):
        for para in (False, True):
            kw = dict(P=P, W=W, method=method)
            R_init = solve(mode="init_R", scaling=scaling, **kw)
            factor = solve(mode="merge_factor", **kw)
            if para:
                dW = solve(mode="delta_from_factor", factor=factor, para=True, **kw)
                assert torch.count_nonzero(dW).item() > 0, (method, "para delta nonzero")
            else:
                dW = solve(mode="delta_from_factor", factor=factor, R=R_init,
                           scaling=scaling, para=False, **kw)
                # identity init: delta == 0 exactly
                assert torch.count_nonzero(dW).item() == 0, (method, "identity init")
                # with a random R the delta matches the closed form
                R = torch.randn(r, in_f)
                dW2 = solve(mode="delta_from_factor", factor=factor, R=R,
                            scaling=scaling, para=False, **kw)
                proj = solve(mode="project", P=P, method=method)
                Q_P, right = proj["Q_P"], proj["right"]
                want = Q_P @ (R.to(torch.float32) * scaling - factor)
                assert torch.allclose(dW2, want, atol=1e-12, rtol=0)
    # para delta equals -P_proj @ W
    proj = solve(mode="project", P=P, method="qr")
    Q_P, right = proj["Q_P"], proj["right"]
    factor = solve(mode="merge_factor", P=P, W=W, method="qr")
    dWp = solve(mode="delta_from_factor", P=P, factor=factor, method="qr", para=True)
    assert torch.allclose(dWp, -(Q_P @ right.T @ W.to(torch.float32)), atol=1e-5, rtol=0)
    # anchor
    assert abs(factor[0, 0].item() - 1.2625722885131836) < 1e-5, factor[0, 0].item()


def check_branch(solve):
    torch.manual_seed(2)
    bs, q, out_f, in_f, r = 2, 5, 6, 8, 3
    scaling = 0.5
    W = torch.randn(out_f, in_f)
    P = torch.randn(out_f, r) * 0.1
    R = torch.randn(r, in_f)
    x = torch.randn(bs, q, in_f)
    bias = torch.randn(out_f)
    base_out = x @ W.T + bias
    base_product = base_out - bias  # x @ W.T

    for method in ("qr", "relu"):
        out = solve(mode="branch", x=x, base_product=base_product, P=P, R=R,
                    scaling=scaling, method=method, para=False)
        # branch == base_product + x @ delta.T (merge/forward consistency)
        factor = solve(mode="merge_factor", P=P, W=W, method=method)
        dW = solve(mode="delta_from_factor", P=P, factor=factor, R=R,
                   scaling=scaling, method=method, para=False)
        assert torch.allclose(out, base_product.to(torch.float32) + x.to(torch.float32) @ dW.T,
                              atol=1e-5, rtol=0), method
        # manual: base_product + [(x @ R.T * scaling) - (x @ W.T) @ right] @ Q_P.T
        proj = solve(mode="project", P=P, method=method)
        Q_P, right = proj["Q_P"], proj["right"]
        want = base_product.to(torch.float32) + ((x.to(torch.float32) @ (R.to(torch.float32) * scaling).T)
                - (x.to(torch.float32) @ W.to(torch.float32).T) @ right) @ Q_P.T
        assert torch.allclose(out, want, atol=1e-5, rtol=0), method

    # para: only the removal term (R ignored)
    out_p = solve(mode="branch", x=x, base_product=base_product, P=P, R=R,
                  scaling=scaling, method="qr", para=True)
    proj2 = solve(mode="project", P=P, method="qr")
    Q_P, right = proj2["Q_P"], proj2["right"]
    want_p = base_product.to(torch.float32) - (base_product.to(torch.float32) @ right) @ Q_P.T
    assert torch.allclose(out_p, want_p, atol=1e-5, rtol=0)
    # para == x @ para_delta.T
    factor = solve(mode="merge_factor", P=P, W=W, method="qr")
    dWp = solve(mode="delta_from_factor", P=P, factor=factor, method="qr", para=True)
    assert torch.allclose(out_p, base_product.to(torch.float32) + x.to(torch.float32) @ dWp.T,
                          atol=1e-5, rtol=0)
    # identity init => the adapted output equals the plain base output
    R_init = solve(mode="init_R", P=P, W=W, method="qr", scaling=scaling)
    out0 = solve(mode="branch", x=x, base_product=base_product, P=P, R=R_init,
                 scaling=scaling, method="qr", para=False)
    assert torch.allclose(out0, base_product.to(torch.float32), atol=1e-5, rtol=0)
    # anchor
    assert abs(out[0, 0, 0].item() - 4.64639949798584) < 1e-5, out[0, 0, 0].item()


CHECKS = [
    "check_resolve",
    "check_project",
    "check_delta_identity",
    "check_branch",
]


def check(solve):
    for name in CHECKS:
        globals()[name](solve)
