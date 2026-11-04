"""Arrow test cases. Judging convention: string CHECKS + check(solve)."""
import torch

torch.set_default_dtype(torch.float64)


def _adapter_pair(r, f_in, f_out, seed):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(r, f_in, generator=g)
    B = torch.randn(f_out, r, generator=g)
    return A, B


def check_singular_vec(solve):
    r, f_in, f_out = 3, 8, 6
    A, B = _adapter_pair(r, f_in, f_out, 0)
    v = solve(mode="singular_vec", a=A, b=B, seed=42)
    assert v.shape == (f_in,) and v.dtype == torch.float32
    # unit norm (up to the eps slack)
    assert abs(v.norm().item() - 1.0) < 1e-5
    # it IS the top right singular vector: rayleigh quotient == top eigenvalue
    dWt_dW = (B @ A).t() @ (B @ A)
    U, S, Vh = torch.linalg.svd((B @ A).double())
    lam = (v.double() @ dWt_dW @ v.double()).item()
    assert abs(lam - S[0].item() ** 2) / S[0].item() ** 2 < 1e-5, (lam, S[0].item() ** 2)
    # rayleigh quotient dominates every other direction sampled
    g = torch.Generator().manual_seed(7)
    for _ in range(5):
        u = torch.randn(f_in, generator=g)
        u = u / u.norm()
        assert (u @ dWt_dW @ u).item() <= lam * (1 + 1e-9) + 1e-9
    # sign-agnostic: -v is the same singular vector
    lam_neg = (-v.double()) @ dWt_dW @ (-v.double())
    assert abs(lam_neg.item() - lam) < 1e-6
    # the seed fixes v0: same seed reproduces, different seed converges to ±v
    v2 = solve(mode="singular_vec", a=A, b=B, seed=42)
    assert torch.equal(v, v2), "seeded power iteration not reproducible"
    v3 = solve(mode="singular_vec", a=A, b=B, seed=7)
    assert (torch.allclose(v3.double().abs(), v.double().abs(), atol=1e-4)
            or torch.allclose(v3.double(), -v.double(), atol=1e-4)
            or torch.allclose(v3.double(), v.double(), atol=1e-4)), \
        "different seeds must converge to the same (up to sign) vector"
    # anchors
    assert abs(v[0].item() - 0.6975467205047607) < 1e-7, v[0].item()
    assert abs(v[-1].item() - 0.2487221509218216) < 1e-7, v[-1].item()


def check_prototypes(solve):
    pairs = [_adapter_pair(3, 8, 6, s) for s in (0, 1, 2)]
    protos = solve(mode="prototypes", pairs=pairs, seed=42)
    assert protos.shape == (3, 8), protos.shape
    # official: the fp32 iteration result is cast back to the LoRA (A) dtype
    assert protos.dtype == pairs[0][0].dtype, protos.dtype
    # each row is the per-adapter singular vector (cast back to the LoRA dtype)
    v0 = solve(mode="singular_vec", a=pairs[0][0], b=pairs[0][1], seed=42)
    assert torch.equal(protos[0], v0.to(pairs[0][0].dtype))
    # distinct adapters yield distinct prototypes
    assert not torch.allclose(protos[0], protos[1], atol=1e-3)
    # anchors
    assert abs(protos[2, 3].item() - (-0.170495867729187)) < 1e-5, protos[2, 3].item()


def check_route_coeffs(solve):
    torch.manual_seed(3)
    t, e = 5, 4
    protos = torch.randn(e, 8)
    tok = torch.randn(t, 8)
    coeff, idx = solve(mode="route_coeffs", tok=tok, prototypes=protos,
                       top_k=2, temperature=1.0)
    assert coeff.shape == (t, e) and idx.shape == (t, 2)
    # non-selected experts have coefficient exactly 0, selected sum to 1
    mask = torch.zeros(t, e, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    assert torch.equal(coeff[~mask], torch.zeros_like(coeff[~mask])), \
        "non-top-k coefficients must be exactly 0"
    assert torch.allclose(coeff.sum(1), torch.ones(t), atol=1e-9)
    # manual account: sim = abs, softmax over the selected scores only
    sim = torch.abs(tok @ protos.t())
    top_v, top_i = torch.topk(sim, 2, dim=1)
    want = torch.zeros(t, e)
    want.scatter_(1, top_i, torch.softmax(top_v, dim=1))
    assert torch.allclose(coeff, want, atol=1e-9), "routing account mismatch"
    # anchors
    assert abs(coeff[0, idx[0, 0]].item() - 0.7688820194964578) < 1e-9
    # temperature reshapes the distribution: T->0 approaches argmax one-hot
    coeff_hot, _ = solve(mode="route_coeffs", tok=tok, prototypes=protos,
                         top_k=2, temperature=0.01)
    assert torch.allclose(coeff_hot.max(1).values, torch.ones(t), atol=1e-3), \
        "low temperature must approach one-hot routing"
    # equal scores split the mass evenly between the two selected experts
    tok_eq = torch.zeros(1, 8)
    protos_eq = torch.randn(4, 8, generator=torch.Generator().manual_seed(5))
    coeff_eq, _ = solve(mode="route_coeffs", tok=tok_eq, prototypes=protos_eq,
                        top_k=2, temperature=1.0)
    sel = coeff_eq[coeff_eq > 0]
    assert torch.allclose(sel, torch.full_like(sel, 0.5), atol=1e-9), \
        "equal scores must split evenly (all sims are 0 for the zero token)"


def check_expert_outputs_and_delta(solve):
    torch.manual_seed(4)
    t, e, r, f_in, f_out = 6, 3, 3, 8, 6
    pairs = [_adapter_pair(r, f_in, f_out, s) for s in (0, 1, 2)]
    A_stack = torch.stack([p[0] for p in pairs])
    B_stack = torch.stack([p[1] for p in pairs])
    scales = torch.tensor([1.0, 2.0, 0.5])
    protos = solve(mode="prototypes", pairs=pairs, seed=42)
    x = torch.randn(2, 3, f_in)
    y = solve(mode="expert_outputs", tok=x.reshape(-1, f_in), a_stack=A_stack,
              b_stack=B_stack, scales=scales)
    assert y.shape == (t, e, f_out), y.shape
    # manual per-expert account with scaling
    for e_i in range(e):
        want = (x.reshape(-1, f_in) @ pairs[e_i][0].t()) @ pairs[e_i][1].t()
        assert torch.allclose(y[:, e_i, :], want * scales[e_i], atol=1e-9)
    delta = solve(mode="arrow_delta", x=x, prototypes=protos, a_stack=A_stack,
                  b_stack=B_stack, scales=scales, top_k=2, temperature=1.0)
    assert delta.shape == (2, 3, f_out), delta.shape
    # full manual account
    coeff, idx = solve(mode="route_coeffs", tok=x.reshape(-1, f_in),
                       prototypes=protos, top_k=2, temperature=1.0)
    want = torch.einsum("t e, t e o -> t o", coeff, y)
    assert torch.allclose(delta.view(t, f_out), want, atol=1e-9), "delta account"
    # anchors
    assert abs(delta.view(t, f_out)[0, 0].item() - 2.7192971987121317) < 1e-9
    # a bigger library changes the routing but keeps the shape
    delta_k1 = solve(mode="arrow_delta", x=x, prototypes=protos, a_stack=A_stack,
                     b_stack=B_stack, scales=scales, top_k=1, temperature=1.0)
    assert delta_k1.shape == (2, 3, f_out)
    assert not torch.allclose(delta_k1, delta, atol=1e-6), "top_k has no effect"
    # scaling is per-adapter: zeroing one scale removes its contribution
    scales0 = torch.tensor([0.0, 2.0, 0.5])
    delta0 = solve(mode="arrow_delta", x=x, prototypes=protos, a_stack=A_stack,
                   b_stack=B_stack, scales=scales0, top_k=2, temperature=1.0)
    coeff0, idx0 = solve(mode="route_coeffs", tok=x.reshape(-1, f_in),
                         prototypes=protos, top_k=2, temperature=1.0)
    y0 = solve(mode="expert_outputs", tok=x.reshape(-1, f_in), a_stack=A_stack,
               b_stack=B_stack, scales=scales0)
    assert torch.allclose(delta0.view(t, f_out),
                          torch.einsum("t e, t e o -> t o", coeff0, y0), atol=1e-9)


def check_compat(solve):
    mods, r = solve(mode="check_compat", ranks=[4, 4],
                    shapes=[((4, 8), (6, 4)), ((4, 8), (6, 4))],
                    module_sets=[{"q", "v"}, {"v", "q"}])
    assert r == 4 and mods == ["q", "v"]
    # rank mismatch raises (shapes identical so the rank check is the only guard)
    try:
        solve(mode="check_compat", ranks=[4, 8],
              shapes=[((4, 8), (6, 4)), ((4, 8), (6, 4))],
              module_sets=[{"q"}, {"q"}])
        raise AssertionError("expected ValueError for rank mismatch")
    except ValueError:
        pass
    # shape mismatch raises
    try:
        solve(mode="check_compat", ranks=[4, 4],
              shapes=[((4, 8), (6, 4)), ((4, 9), (6, 4))],
              module_sets=[{"q"}, {"q"}])
        raise AssertionError("expected ValueError for shape mismatch")
    except ValueError:
        pass
    # target_modules mismatch raises
    try:
        solve(mode="check_compat", ranks=[4, 4],
              shapes=[((4, 8), (6, 4)), ((4, 8), (6, 4))],
              module_sets=[{"q", "v"}, {"q"}])
        raise AssertionError("expected ValueError for module mismatch")
    except ValueError:
        pass


CHECKS = [
    "check_singular_vec",
    "check_prototypes",
    "check_route_coeffs",
    "check_expert_outputs_and_delta",
    "check_compat",
]


def check(solve):
    for name in CHECKS:
        globals()[name](solve)
