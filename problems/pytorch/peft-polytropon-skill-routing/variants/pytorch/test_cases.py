"""Polytropon test cases — judge-style: check(solve) + string CHECKS."""
import math

import torch

torch.set_default_dtype(torch.float64)


def check_init_params(solve):
    ns, sk, in_f, out_f, r = 2, 3, 8, 6, 2
    f = solve(mode="init_params", n_splits=ns, n_skills=sk, in_features=in_f,
              out_features=out_f, r=r, seed=0, n_tasks=4, init_weights=True)
    A, B, logits = f["A"], f["B"], f["module_logits"]
    assert A.shape == (ns, sk, in_f // ns, r), A.shape
    assert B.shape == (ns, sk, r, out_f // ns), B.shape
    assert logits.shape == (4, ns * sk), logits.shape
    assert torch.count_nonzero(B).item() == 0, "init_weights=True => B zeros"
    # RNG domain: skill-outer, split-inner kaiming on (r, d) then transposed.
    # fan_in = d = in // n_splits; a=sqrt(5) => bound = sqrt(6/((1+a^2) d)).
    # Use the exact same float formula as the solution (sqrt(5)**2 is not exactly 5).
    a = math.sqrt(5)
    bound = math.sqrt(6.0 / ((1.0 + a * a) * (in_f // ns)))
    g = torch.Generator().manual_seed(0)
    for skill in range(sk):
        for split in range(ns):
            ref = torch.empty(r, in_f // ns, dtype=torch.float64).uniform_(
                -bound, bound, generator=g).T
            assert torch.equal(A[split, skill], ref), (skill, split)
    # router logits consume the generator AFTER all A blocks
    ref_logits = torch.empty(4, ns * sk, dtype=torch.float64).uniform_(
        -1e-3, 1e-3, generator=g)
    assert torch.equal(logits, ref_logits)
    # init_weights=False: B kaiming on (d_out, r) transposed, fan_in = r
    f2 = solve(mode="init_params", n_splits=ns, n_skills=sk, in_features=in_f,
               out_features=out_f, r=r, seed=0, n_tasks=4, init_weights=False)
    g2 = torch.Generator().manual_seed(0)
    for skill in range(sk):
        for split in range(ns):
            torch.empty(r, in_f // ns, dtype=torch.float64).uniform_(
                -bound, bound, generator=g2)  # A blocks consume the stream first
    bbound = math.sqrt(6.0 / ((1.0 + a * a) * r))
    for skill in range(sk):
        for split in range(ns):
            ref = torch.empty(out_f // ns, r, dtype=torch.float64).uniform_(
                -bbound, bbound, generator=g2).T
            assert torch.equal(f2["B"][split, skill], ref), (skill, split)


def check_route(solve):
    torch.manual_seed(0)
    ns, sk, n_tasks = 2, 3, 4
    logits = torch.randn(n_tasks, ns * sk)
    task_ids = torch.tensor([1, 0, 3])
    w = solve(mode="route", task_ids=task_ids, module_logits=logits,
              n_splits=ns, n_skills=sk)
    assert w.shape == (3, ns, sk), w.shape
    s = torch.sigmoid(logits[[1, 0, 3]].view(3, ns, sk))
    want = s / (s.sum(dim=-1, keepdim=True) + 1e-12)
    assert torch.allclose(w, want, atol=1e-12, rtol=0)
    # rows sum to ~1 per split (normalized sigmoids, not softmax)
    sums = w.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-9)
    # sigmoid normalization differs from softmax
    exp = torch.exp(logits[[1, 0, 3]].view(3, ns, sk))
    softmax_w = exp / exp.sum(dim=-1, keepdim=True)
    assert not torch.allclose(w, softmax_w, atol=1e-6)
    # invalid task id raises
    try:
        solve(mode="route", task_ids=torch.tensor([4]), module_logits=logits,
              n_splits=ns, n_skills=sk)
        raise AssertionError("expected ValueError for out-of-range task id")
    except ValueError:
        pass
    # EPS guard: with sigmoids underflowed to 0, the +EPS keeps the division finite
    tiny = torch.full((n_tasks, ns * sk), -800.0)
    w_t = solve(mode="route", task_ids=torch.tensor([0]), module_logits=tiny,
                n_splits=ns, n_skills=sk)
    assert torch.isfinite(w_t).all() and torch.count_nonzero(w_t).item() == 0
    # anchor
    assert abs(w[0, 0, 0].item() - 0.28980108513541014) < 1e-12, w[0, 0, 0].item()


def check_combine(solve):
    torch.manual_seed(1)
    bs, ns, sk, in_f, out_f, r = 2, 2, 3, 8, 6, 2
    w = torch.rand(bs, ns, sk)
    A = torch.randn(ns, sk, in_f // ns, r)
    B = torch.randn(ns, sk, r, out_f // ns)
    out = solve(mode="combine", weights=w, A=A, B=B, r=r)
    A_mixed, B_mixed = out["A_mixed"], out["B_mixed"]
    assert A_mixed.shape == (bs, in_f, r), A_mixed.shape
    assert B_mixed.shape == (bs, r, out_f), B_mixed.shape
    # A reassembly: split 0 occupies input rows [0, in//ns), split 1 the rest
    want_a = torch.einsum("bqs,qsdr->bqdr", w, A).reshape(bs, in_f, r)
    assert torch.equal(A_mixed, want_a)
    # B reassembly: transpose(1,2) then reshape — output cols are split-major
    want_b = torch.einsum("bqs,qsrd->bqrd", w, B).transpose(1, 2).reshape(bs, r, out_f)
    assert torch.equal(B_mixed, want_b)
    # per-split weighting identity: A_mixed[:, :in//ns] only involves split 0
    solo = torch.einsum("bs,sdr->bdr", w[:, 0, :], A[0])
    assert torch.equal(A_mixed[:, : in_f // ns], solo)
    # anchor
    assert abs(A_mixed[0, 0, 0].item() - 0.33493025197789034) < 1e-12, A_mixed[0, 0, 0].item()


def check_forward_delta(solve):
    torch.manual_seed(2)
    bs, q, ns, sk, in_f, out_f, r = 2, 3, 2, 3, 8, 6, 2
    x = torch.randn(bs, q, in_f)
    w = torch.rand(bs, ns, sk)
    A = torch.randn(ns, sk, in_f // ns, r)
    B = torch.randn(ns, sk, r, out_f // ns)
    delta = solve(mode="forward_delta", x=x, weights=w, A=A, B=B, r=r)
    assert delta.shape == (bs, q, out_f), delta.shape
    out = solve(mode="combine", weights=w, A=A, B=B, r=r)
    want = x.bmm(out["A_mixed"]).bmm(out["B_mixed"]) / r
    assert torch.allclose(delta, want, atol=1e-12, rtol=0)
    # scaling is exactly 1/r (not 1/sqrt(r), not alpha/r)
    want_noscale = x.bmm(out["A_mixed"]).bmm(out["B_mixed"])
    assert torch.allclose(delta * r, want_noscale, atol=1e-9, rtol=0)


def check_param_count(solve):
    c = solve(mode="param_count", n_splits=2, n_skills=4, in_features=8,
              out_features=6, r=2, n_tasks=3)
    assert c["lora"] == 2 * 4 * 4 * 2 + 2 * 4 * 2 * 3, c
    assert c["router"] == 3 * 2 * 4, c
    assert c["total"] == c["lora"] + c["router"], c
    # n_splits=1 reduces to plain multi-LoRA inventory
    c1 = solve(mode="param_count", n_splits=1, n_skills=4, in_features=8,
               out_features=6, r=2, n_tasks=3)
    assert c1["lora"] == 4 * 8 * 2 + 4 * 2 * 6, c1


CHECKS = [
    "check_init_params",
    "check_route",
    "check_combine",
    "check_forward_delta",
    "check_param_count",
]


def check(solve):
    for name in CHECKS:
        globals()[name](solve)
