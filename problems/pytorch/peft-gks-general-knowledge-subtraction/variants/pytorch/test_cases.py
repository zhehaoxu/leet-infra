"""GenKnowSub test cases. Judging convention: string CHECKS + check(solve)."""
import torch

torch.set_default_dtype(torch.float64)


def _lib(e, r, f_in, f_out, seed):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(e, r, f_in, generator=g)
    b = torch.randn(e, f_out, r, generator=g)
    return a, b


def check_gks_average(solve):
    g_a, g_b = _lib(3, 2, 8, 6, 0)
    avg_a, avg_b = solve(mode="gks_average", g_a=list(g_a), g_b=list(g_b))
    assert avg_a.shape == (2, 8) and avg_b.shape == (6, 2)
    # simple mean over the general adapters
    want_a = g_a.mean(0)
    want_b = g_b.mean(0)
    assert torch.allclose(avg_a, want_a, atol=1e-12)
    assert torch.allclose(avg_b, want_b, atol=1e-12)
    # anchors
    assert abs(avg_a[0, 0].item() - (-0.7495449019423001)) < 1e-12, avg_a[0, 0].item()
    # a single general adapter is its own average
    avg1_a, avg1_b = solve(mode="gks_average", g_a=[g_a[0]], g_b=[g_b[0]])
    assert torch.equal(avg1_a, g_a[0]) and torch.equal(avg1_b, g_b[0])


def check_gks_subtract(solve):
    e_t, r, f_in, f_out = 4, 2, 8, 6
    a, b = _lib(e_t, r, f_in, f_out, 1)
    g_a, g_b = _lib(3, r, f_in, f_out, 0)
    avg_a, avg_b = solve(mode="gks_average", g_a=list(g_a), g_b=list(g_b))
    a_new, b_new = solve(mode="gks_subtract", a_stack=a, b_stack=b,
                         avg_a=avg_a, avg_b=avg_b)
    assert a_new.shape == a.shape and b_new.shape == b.shape
    # every task factor loses exactly the general average
    assert torch.allclose(a_new, a - avg_a, atol=1e-12)
    assert torch.allclose(b_new, b - avg_b, atol=1e-12)
    # anchors
    assert abs(a_new[0, 0, 0].item() - 0.438255347752268) < 1e-12
    # inputs are untouched (pure function over the in-place official op)
    assert torch.allclose(a, _lib(e_t, r, f_in, f_out, 1)[0], atol=1e-15)
    # the "added" subset limits purification to those experts
    a_p, b_p = solve(mode="gks_subtract", a_stack=a, b_stack=b,
                     avg_a=avg_a, avg_b=avg_b, only=[1, 3])
    assert torch.allclose(a_p[1], a[1] - avg_a, atol=1e-12)
    assert torch.allclose(a_p[3], a[3] - avg_a, atol=1e-12)
    assert torch.equal(a_p[0], a[0]) and torch.equal(a_p[2], a[2])
    assert torch.equal(b_p[0], b[0]) and torch.equal(b_p[2], b[2])
    # an unsorted subset purifies the same experts (official name list order)
    a_u, b_u = solve(mode="gks_subtract", a_stack=a, b_stack=b,
                     avg_a=avg_a, avg_b=avg_b, only=[3, 1])
    assert torch.allclose(a_u, a_p, atol=1e-12)
    assert torch.allclose(b_u, b_p, atol=1e-12)


def check_gks_gate(solve):
    assert solve(mode="gks_gate", use_gks=False, gks_done=False, n_added=0) == "skip"
    assert solve(mode="gks_gate", use_gks=True, gks_done=False, n_added=0) == "all"
    assert solve(mode="gks_gate", use_gks=True, gks_done=True, n_added=0) == "skip"
    assert solve(mode="gks_gate", use_gks=True, gks_done=True, n_added=2) == "added"
    # use_gks wins over everything
    assert solve(mode="gks_gate", use_gks=False, gks_done=True, n_added=3) == "skip"
    # anchors
    assert solve(mode="gks_gate", use_gks=True, gks_done=False, n_added=5) == "all"


def check_require_paths(solve):
    assert solve(mode="require_gks_paths", use_gks=True, n_paths=2) == 2
    assert solve(mode="require_gks_paths", use_gks=False, n_paths=0) == 0
    try:
        solve(mode="require_gks_paths", use_gks=True, n_paths=0)
        raise AssertionError("expected ValueError for missing general paths")
    except ValueError:
        pass


def check_task_delta(solve):
    e_t, r, f_in, f_out = 3, 2, 8, 6
    a, b = _lib(e_t, r, f_in, f_out, 2)
    g_a, g_b = _lib(2, r, f_in, f_out, 0)
    avg_a, avg_b = solve(mode="gks_average", g_a=list(g_a), g_b=list(g_b))
    a_new, b_new = solve(mode="gks_subtract", a_stack=a, b_stack=b,
                         avg_a=avg_a, avg_b=avg_b)
    d = solve(mode="task_delta", a_sub=a_new, b_sub=b_new)
    assert d.shape == (e_t, f_out, f_in), d.shape
    for i in range(e_t):
        want = b_new[i] @ a_new[i]
        assert torch.allclose(d[i], want, atol=1e-9)
    # anchors
    assert abs(d[0, 0, 0].item() - 2.1708577088384393) < 1e-9, d[0, 0, 0].item()
    # factor subtraction carries CROSS terms: (B-gB)@(A-gA) expands to
    # B@A - B@gA - gB@A + gB@gA, NOT B@A - gB@gA
    plain = torch.einsum("e o r, e r f -> e o f", b, a)
    cross = (torch.einsum("o r, r f -> o f", b[0], avg_a)
             + torch.einsum("o r, r f -> o f", avg_b, a[0])
             - torch.einsum("o r, r f -> o f", avg_b, avg_a))
    assert torch.allclose(d[0], plain[0] - cross, atol=1e-9)
    naive = plain[0] - torch.einsum("o r, r f -> o f", avg_b, avg_a)
    assert not torch.allclose(d[0], naive, atol=1e-6), "cross terms vanished"


CHECKS = [
    "check_gks_average",
    "check_gks_subtract",
    "check_gks_gate",
    "check_require_paths",
    "check_task_delta",
]


def check(solve):
    for name in CHECKS:
        globals()[name](solve)
