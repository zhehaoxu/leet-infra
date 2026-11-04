"""OSF test cases — judge-style: check(solve) + string CHECKS."""
import math

import torch

torch.set_default_dtype(torch.float64)


def check_decompose(solve):
    torch.manual_seed(0)
    out_f, in_f = 6, 8
    W = torch.randn(out_f, in_f)
    d = solve(mode="decompose", weight=W, top_k=4)
    assert d["U_high"].shape == (out_f, 4), d["U_high"].shape
    assert d["S_high"].shape == (4,), d["S_high"].shape
    assert d["V_high"].shape == (4, in_f), d["V_high"].shape
    assert d["U_low"].shape == (out_f, 2), d["U_low"].shape
    assert d["S_low"].shape == (2,), d["S_low"].shape
    assert d["V_low"].shape == (2, in_f), d["V_low"].shape
    assert d["rank_high"] == 4
    # top_k clamped to min_dim
    d2 = solve(mode="decompose", weight=W, top_k=100)
    assert d2["rank_high"] == 6 and d2["U_low"].shape == (out_f, 0)
    # high part is the truncated SVD (matches torch.linalg.svd head)
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    assert torch.allclose(d["U_high"], U[:, :4], atol=1e-9)
    assert torch.allclose(d["S_high"], S[:4], atol=1e-9)
    assert torch.allclose(d["V_high"], Vh[:4], atol=1e-9)
    assert torch.allclose(d["S_low"], S[4:], atol=1e-9)
    # anchors
    assert abs(d["U_high"][0, 0].item() - 0.4377374053001404) < 1e-9, d["U_high"][0, 0].item()
    assert abs(d["S_high"][0].item() - 4.716036319732666) < 1e-9, d["S_high"][0].item()


def check_reconstruct(solve):
    torch.manual_seed(1)
    out_f, in_f = 5, 7
    W = torch.randn(out_f, in_f)
    d = solve(mode="decompose", weight=W, top_k=3)
    W_rec = solve(mode="reconstruct", svd=d)
    assert W_rec.shape == W.shape
    # exact inverse: high + low == W (float32 SVD pipeline => 1e-5)
    assert torch.allclose(W_rec, W, atol=1e-5, rtol=0), (W_rec - W).abs().max()
    # high alone is the best rank-3 approximation
    high = d["U_high"] @ torch.diag(d["S_high"]) @ d["V_high"]
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    assert torch.allclose(high, U[:, :3] @ torch.diag(S[:3]) @ Vh[:3], atol=1e-5, rtol=0)
    # anchor (float32 pipeline)
    assert abs(W_rec[0, 0].item() - W[0, 0].item()) < 1e-5


def check_resolve_rank(solve):
    # int passes through
    assert solve(mode="resolve_rank", effective_rank=4, min_dim=8) == 4
    # None -> 50% of min_dim
    assert solve(mode="resolve_rank", effective_rank=None, min_dim=8) == 4
    assert solve(mode="resolve_rank", effective_rank=None, min_dim=7) == 3
    # float fraction of min_dim
    assert solve(mode="resolve_rank", effective_rank=0.8, min_dim=10) == 8
    assert solve(mode="resolve_rank", effective_rank=0.5, min_dim=7) == 3
    # out-of-range fraction raises
    try:
        solve(mode="resolve_rank", effective_rank=1.5, min_dim=8)
        raise AssertionError("expected ValueError for fraction > 1")
    except ValueError:
        pass
    try:
        solve(mode="resolve_rank", effective_rank=0.0, min_dim=8)
        raise AssertionError("expected ValueError for fraction == 0")
    except ValueError:
        pass


def check_project_u(solve):
    torch.manual_seed(2)
    out_f, k = 6, 3
    U_high = torch.linalg.qr(torch.randn(out_f, k))[0]  # orthonormal columns, like SVD U
    dU = torch.randn(out_f, out_f - k)
    proj = solve(mode="project_u", grad=dU, U_high=U_high)
    assert proj.shape == dU.shape
    want = dU - U_high @ (U_high.T @ dU)
    assert torch.allclose(proj, want, atol=1e-12, rtol=0)
    # projected gradient is orthogonal to U_high's column space
    assert torch.allclose(U_high.T @ proj, torch.zeros(k, proj.shape[1]), atol=1e-9), (U_high.T @ proj).abs().max()
    # already-orthogonal input is unchanged
    Q, _ = torch.linalg.qr(torch.randn(out_f, k))
    ortho = Q @ torch.randn(k, 2)
    # remove U_high component first so it lives in the complement
    ortho = ortho - Q @ (Q.T @ ortho)
    p2 = solve(mode="project_u", grad=ortho, U_high=U_high)
    assert torch.allclose(p2, ortho, atol=1e-9)
    # a gradient lying fully inside U_high's span projects to zero
    inside = U_high @ torch.randn(k, 2)
    p3 = solve(mode="project_u", grad=inside, U_high=U_high)
    assert torch.count_nonzero(p3).item() <= 0 or torch.allclose(p3, torch.zeros_like(p3), atol=1e-9)


def check_project_v(solve):
    torch.manual_seed(3)
    k, in_f = 3, 8
    V_high = torch.linalg.qr(torch.randn(in_f, k))[0].T  # orthonormal rows, like SVD Vh
    dV = torch.randn(4, in_f)
    proj = solve(mode="project_v", grad=dV, V_high=V_high)
    assert proj.shape == dV.shape
    G = V_high.T @ V_high
    want = dV - dV @ G
    assert torch.allclose(proj, want, atol=1e-12, rtol=0)
    # rows of the projection are orthogonal to V_high's rows
    assert torch.allclose(proj @ V_high.T, torch.zeros(proj.shape[0], k), atol=1e-9), (proj @ V_high.T).abs().max()
    # anchor
    assert abs(proj[0, 0].item() - (-0.34366725892785327)) < 1e-9, proj[0, 0].item()


CHECKS = [
    "check_decompose",
    "check_reconstruct",
    "check_resolve_rank",
    "check_project_u",
    "check_project_v",
]


def check(solve):
    for name in CHECKS:
        globals()[name](solve)
