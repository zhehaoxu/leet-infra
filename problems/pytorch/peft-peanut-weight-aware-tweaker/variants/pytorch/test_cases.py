"""PEANuT test cases — judge-style: check(solve) + string CHECKS."""
import math

import torch

CHECKS = ["check_delta_weight", "check_forward", "check_param_count",
          "check_init_factors", "check_residual_order"]


def check_delta_weight(solve):
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    out_f, in_f, r, depth = 6, 8, 3, 2
    W = torch.randn(out_f, in_f)
    A = torch.randn(r, out_f)
    enc = [torch.randn(r, r) for _ in range(depth)]
    dec = [torch.randn(r, r) for _ in range(depth)]
    B = torch.randn(out_f, r)

    delta = solve(mode="delta_weight", weight=W, a=A, encoders=enc, decoders=dec,
                  b=B, scaling=2.0, act_fn="relu")
    assert delta.shape == (out_f, in_f), delta.shape

    # manual replay of the official loop (residual reverse order)
    d = torch.relu(W.t() @ A.t())
    res = []
    for e in enc:
        res.append(d)
        d = torch.relu(d @ e.t())
    for i, dc in enumerate(dec):
        d = torch.relu(d @ dc.t()) + res[depth - 1 - i]
    want = (d @ B.t() * 2.0).t()
    assert torch.allclose(delta, want, atol=1e-9, rtol=0), (delta - want).abs().max()

    # depth=0 path: act(W^T A^T) @ B^T
    delta0 = solve(mode="delta_weight", weight=W, a=A, encoders=[], decoders=[],
                   b=B, scaling=1.0, act_fn="relu")
    want0 = (torch.relu(W.t() @ A.t()) @ B.t()).t()
    assert torch.allclose(delta0, want0, atol=1e-9, rtol=0)

    # depth=1 with tanh: residuals saved before encoder, added after decoder act
    delta1 = solve(mode="delta_weight", weight=W, a=A, encoders=enc[:1], decoders=dec[:1],
                   b=B, scaling=0.5, act_fn="tanh")
    d1 = torch.tanh(W.t() @ A.t())
    res0 = d1
    d1 = torch.tanh(d1 @ enc[0].t())
    d1 = torch.tanh(d1 @ dec[0].t()) + res0
    want1 = (d1 @ B.t() * 0.5).t()
    assert torch.allclose(delta1, want1, atol=1e-9, rtol=0), (delta1 - want1).abs().max()

    # gelu / silu accepted
    for fn in ("gelu", "silu", "identity"):
        out = solve(mode="delta_weight", weight=W, a=A, encoders=enc[:1], decoders=dec[:1],
                    b=B, act_fn=fn)
        assert out.shape == (out_f, in_f)

    # unsupported act raises
    try:
        solve(mode="delta_weight", weight=W, a=A, encoders=[], decoders=[], b=B, act_fn="swiglu")
        raise AssertionError("expected ValueError for bad act_fn")
    except ValueError:
        pass


def check_forward(solve):
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(1)
    out_f, in_f, r, depth = 5, 7, 3, 1
    W = torch.randn(out_f, in_f)
    A = torch.randn(r, out_f)
    enc = [torch.randn(r, r)]
    dec = [torch.randn(r, r)]
    B = torch.randn(out_f, r)
    x = torch.randn(4, in_f)

    y = solve(mode="forward", x=x, weight=W, a=A, encoders=enc, decoders=dec,
              b=B, scaling=1.5, act_fn="relu")
    assert y.shape == (4, out_f), y.shape

    d = torch.relu(W.t() @ A.t())
    res = [d]
    d = torch.relu(d @ enc[0].t())
    d = torch.relu(d @ dec[0].t()) + res[0]
    delta = (d @ B.t() * 1.5).t()
    want = x @ W.t() + x @ delta.t()
    assert torch.allclose(y, want, atol=1e-9, rtol=0), (y - want).abs().max()


def check_param_count(solve):
    counts = solve(mode="param_count", in_features=1024, out_features=256, r=8, depth=1)
    assert counts["lora"] == 1024 * 8 + 8 * 256, counts
    assert counts["peanut"] == 256 * 8 + 8 * 256 + 2 * 1 * 8 * 8, counts
    assert counts["saving"] == counts["lora"] - counts["peanut"], counts
    # down_proj-style layer: in > out => PEANuT strictly cheaper
    # saving = r*(in-out) - 2*depth*r^2
    assert counts["saving"] == 8 * (1024 - 256) - 128, counts

    counts0 = solve(mode="param_count", in_features=100, out_features=100, r=4, depth=0)
    assert counts0["peanut"] == counts0["lora"], counts0
    assert counts0["saving"] == 0, counts0

    counts2 = solve(mode="param_count", in_features=64, out_features=64, r=4, depth=2)
    assert counts2["peanut"] == 64 * 4 + 4 * 64 + 2 * 2 * 4 * 4, counts2


def check_init_factors(solve):
    torch.set_default_dtype(torch.float64)
    f = solve(mode="init_factors", in_features=8, out_features=6, r=3, depth=1, seed=0)
    assert f["A"].shape == (3, 6)
    assert len(f["encoders"]) == 1 and f["encoders"][0].shape == (3, 3)
    assert len(f["decoders"]) == 1 and f["decoders"][0].shape == (3, 3)
    assert torch.count_nonzero(f["B"]).item() == 0, "init_weights=True => zero B"

    # kaiming_uniform_(a=sqrt(5)) bound: uniform(-1/sqrt(fan_in), 1/sqrt(fan_in))
    assert f["A"].abs().max().item() <= 1.0 / (6 ** 0.5) + 1e-12
    assert f["encoders"][0].abs().max().item() <= 1.0 / (3 ** 0.5) + 1e-12

    # deterministic replay: A first, then encoders, then decoders
    g = torch.Generator().manual_seed(0)
    a_ref = torch.empty(3, 6, dtype=torch.float64).uniform_(-1.0 / math.sqrt(6), 1.0 / math.sqrt(6), generator=g)
    e_ref = torch.empty(3, 3, dtype=torch.float64).uniform_(-1.0 / math.sqrt(3), 1.0 / math.sqrt(3), generator=g)
    d_ref = torch.empty(3, 3, dtype=torch.float64).uniform_(-1.0 / math.sqrt(3), 1.0 / math.sqrt(3), generator=g)
    assert torch.equal(f["A"], a_ref)
    assert torch.equal(f["encoders"][0], e_ref)
    assert torch.equal(f["decoders"][0], d_ref)

    f2 = solve(mode="init_factors", in_features=8, out_features=6, r=3, depth=0, seed=1,
               init_weights=False)
    assert torch.count_nonzero(f2["B"]).item() > 0, "init_weights=False => kaiming B"
    assert f2["B"].abs().max().item() <= 1.0 / (3 ** 0.5) + 1e-12
    assert f2["A"].shape == (3, 6) and f2["encoders"] == [] and f2["decoders"] == []

    # zero-init invariant: B=0 forces delta == 0 regardless of upstream weights
    torch.manual_seed(3)
    Wz = torch.randn(6, 8)
    assert solve(mode="zero_init_check", in_features=8, out_features=6, r=3, depth=1,
                 seed=0, weight=Wz) is True
    assert solve(mode="zero_init_check", in_features=8, out_features=6, r=3, depth=0,
                 seed=0, weight=Wz) is True


def check_residual_order(solve):
    torch.set_default_dtype(torch.float64)
    # pairing must be (i, depth-1-i): reverse-order residual adds
    assert solve(mode="residual_order", depth=0) == []
    assert solve(mode="residual_order", depth=1) == [(0, 0)]
    assert solve(mode="residual_order", depth=2) == [(0, 1), (1, 0)]
    assert solve(mode="residual_order", depth=3) == [(0, 2), (1, 1), (2, 0)]

    # end-to-end proof with constructed weights: swap encoders vs decoders
    # ordering changes the result unless residuals are added in reverse.
    torch.manual_seed(2)
    out_f, in_f, r = 4, 4, 4
    W = torch.randn(out_f, in_f)
    A = torch.randn(r, out_f)
    B = torch.randn(out_f, r)
    enc = [torch.eye(r) * (i + 1) for i in range(2)]
    dec = [torch.eye(r) * (i + 1) for i in range(2)]
    d = solve(mode="delta_weight", weight=W, a=A, encoders=enc, decoders=dec,
              b=B, act_fn="identity")
    # manual: identity act; residuals reverse
    d0 = W.t() @ A.t()
    h = d0 @ enc[0].t()
    res1 = d0  # residual saved before enc0
    h2 = h @ enc[1].t()
    res2 = h   # residual saved before enc1
    o = h2 @ dec[0].t() + res2
    o = o @ dec[1].t() + res1
    want = (o @ B.t()).t()
    assert torch.allclose(d, want, atol=1e-9, rtol=0), (d - want).abs().max()


def check(solve):
    for name in CHECKS:
        globals()[name](solve)
