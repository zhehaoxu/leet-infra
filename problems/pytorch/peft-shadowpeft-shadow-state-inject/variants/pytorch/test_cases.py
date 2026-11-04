"""ShadowPEFT test cases — judge-style: check(solve) + string CHECKS."""
import math

import torch

CHECKS = ["check_inject", "check_update", "check_gated_identity",
          "check_param_count", "check_init_injection"]


def check_inject(solve):
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    b, t, d, k = 2, 3, 8, 4
    hidden = torch.randn(b, t, d)
    shadow = torch.randn(b, t, d)
    w_down = torch.randn(d, k)
    w_up = torch.randn(k, d)

    out = solve(mode="inject", hidden=hidden, shadow=shadow, w_down=w_down,
                w_up=w_up, alpha=0.1)
    assert out.shape == (b, t, d), out.shape

    delta = hidden - shadow
    want = hidden + 0.1 * ((delta @ w_down) @ w_up)
    assert torch.allclose(out, want, atol=1e-9, rtol=0), (out - want).abs().max()

    # alpha=0 disables injection entirely
    out0 = solve(mode="inject", hidden=hidden, shadow=shadow, w_down=w_down,
                 w_up=w_up, alpha=0.0)
    assert torch.equal(out0, hidden), "alpha=0 must be identity"

    # official zero-init invariant: ups zeros => out == hidden
    assert solve(mode="inject_zero_init", hidden=hidden, shadow=shadow,
                 w_down=w_down, w_up=torch.empty(k, d), alpha=0.1) is True

    # sequence: per-layer downs/ups applied in order
    L = 3
    downs = torch.randn(L, d, k)
    ups = torch.randn(L, k, d)
    outs = solve(mode="inject_sequence", hidden=hidden, shadow=shadow,
                 downs=downs, ups=ups, alpha=0.2)
    assert len(outs) == L
    h = hidden
    for l in range(L):
        h = h + 0.2 * (((h - shadow) @ downs[l]) @ ups[l])
        assert torch.allclose(outs[l], h, atol=1e-9, rtol=0), (outs[l] - h).abs().max()

    # bad alpha type irrelevant; but unknown mode raises
    try:
        solve(mode="inject_wrong")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def check_update(solve):
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(1)
    b, t, d, gh = 2, 3, 8, 5
    hidden = torch.randn(b, t, d)
    shadow = torch.randn(b, t, d)
    ln_w = torch.randn(d)
    ln_b = torch.randn(d)
    gate_w1 = torch.randn(gh, d)   # nn.Linear(d -> gh).weight
    gate_w2 = torch.randn(d, gh)   # nn.Linear(gh -> d).weight
    tf_w1 = torch.randn(gh, d)
    tf_w2 = torch.randn(d, gh)

    out = solve(mode="update", hidden=hidden, shadow=shadow, ln_w=ln_w, ln_b=ln_b,
                gate_w1=gate_w1, gate_w2=gate_w2, tf_w1=tf_w1, tf_w2=tf_w2)
    assert out.shape == (b, t, d), out.shape

    h_in = torch.nn.functional.layer_norm(hidden, (d,), ln_w, ln_b, 1e-5)
    g = torch.sigmoid((torch.nn.functional.silu(h_in @ gate_w1.t())) @ gate_w2.t())
    ht = (torch.nn.functional.silu(h_in @ tf_w1.t())) @ tf_w2.t()
    want = shadow + g * (ht - shadow)
    assert torch.allclose(out, want, atol=1e-9, rtol=0), (out - want).abs().max()

    # gate values match the update's gate
    g2 = solve(mode="gate", hidden=hidden, ln_w=ln_w, ln_b=ln_b,
               gate_w1=gate_w1, gate_w2=gate_w2)
    assert torch.allclose(g2, g, atol=1e-12, rtol=0), (g2 - g).abs().max()
    assert bool((g2 > 0).all()) and bool((g2 < 1).all())

    # gate MLP has no dropout and no bias: linear -> silu -> linear -> sigmoid
    g_manual = torch.sigmoid((torch.nn.functional.silu(h_in @ gate_w1.t())) @ gate_w2.t())
    assert torch.allclose(g2, g_manual, atol=1e-12, rtol=0)


def check_gated_identity(solve):
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(2)
    s = torch.randn(2, 3, 8)
    ht = torch.randn(2, 3, 8)
    g = torch.rand(2, 3, 8)
    assert solve(mode="gated_identity", shadow=s, gate=g, ht=ht) is True

    # algebraic equivalence spot-check
    left = s + g * (ht - s)
    right = (1 - g) * s + g * ht
    assert torch.allclose(left, right, atol=1e-12, rtol=0)


def check_param_count(solve):
    counts = solve(mode="param_count", num_layers=28, hidden_size=1024,
                   injection_hidden_size=16, gate_hidden_size=10)
    assert counts["injection"] == 28 * (1024 * 16 + 16 * 1024), counts
    assert counts["update"] == 28 * 2 * (1024 * 10 + 10 * 1024) + 2 * 1024, counts
    assert counts["total"] == counts["injection"] + counts["update"], counts

    counts1 = solve(mode="param_count", num_layers=1, hidden_size=64,
                    injection_hidden_size=8, gate_hidden_size=4)
    assert counts1["injection"] == 1024, counts1
    assert counts1["update"] == 2 * (64 * 4 + 4 * 64) * 1 + 128, counts1


def check_init_injection(solve):
    torch.set_default_dtype(torch.float64)
    f = solve(mode="init_injection", num_layers=3, hidden_size=8,
              injection_hidden_size=4, seed=0)
    assert f["downs"].shape == (3, 8, 4)
    assert f["ups"].shape == (3, 4, 8)
    assert torch.count_nonzero(f["ups"]).item() == 0, "official ups init is zeros"
    assert abs(f["downs"].std().item() - 0.02) < 5e-3, f["downs"].std()

    # deterministic replay: single normal_ draw of the full (L, d, k) block
    g = torch.Generator().manual_seed(0)
    ref = torch.empty(3, 8, 4, dtype=torch.float64).normal_(mean=0.0, std=0.02, generator=g)
    assert torch.equal(f["downs"], ref)

    # custom std respected
    f2 = solve(mode="init_injection", num_layers=1, hidden_size=6,
               injection_hidden_size=3, seed=1, initialization_std=0.5)
    ref2 = torch.empty(1, 6, 3, dtype=torch.float64).normal_(mean=0.0, std=0.5,
                                                             generator=torch.Generator().manual_seed(1))
    assert torch.equal(f2["downs"], ref2)


def check(solve):
    for name in CHECKS:
        globals()[name](solve)
