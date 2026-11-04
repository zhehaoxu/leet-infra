"""add_weighted_adapter test cases — judge-style: check(solve) + string CHECKS."""
import math

import torch

torch.set_default_dtype(torch.float64)


def check_resolve(solve):
    # single source adapter forces linear regardless of the requested type
    out = solve(mode="resolve_rank", ranks=[8], combination_type="cat", svd_rank=2)
    assert out == {"combination_type": "linear", "new_rank": 8}, out
    # linear family: equal ranks required
    out = solve(mode="resolve_rank", ranks=[4, 4], combination_type="linear")
    assert out == {"combination_type": "linear", "new_rank": 4}, out
    try:
        solve(mode="resolve_rank", ranks=[4, 8], combination_type="linear")
        raise AssertionError("expected ValueError for unequal ranks")
    except ValueError:
        pass
    # cat: sum of ranks
    assert solve(mode="resolve_rank", ranks=[4, 8], combination_type="cat")["new_rank"] == 12
    # svd family: svd_rank wins, else max of ranks
    assert solve(mode="resolve_rank", ranks=[4, 8], combination_type="svd", svd_rank=6)["new_rank"] == 6
    assert solve(mode="resolve_rank", ranks=[4, 8], combination_type="svd")["new_rank"] == 8
    assert solve(mode="resolve_rank", ranks=[4, 8], combination_type="ties_svd")["new_rank"] == 8
    # invalid combination type
    try:
        solve(mode="resolve_rank", ranks=[4, 8], combination_type="nope")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    # per-adapter allocation rank with rank_pattern
    assert solve(mode="effective_rank", r=4) == 4
    assert solve(mode="effective_rank", r=4, rank_pattern={"a": 7, "b": 2}) == 7
    assert solve(mode="effective_rank", r=9, rank_pattern={"a": 7}) == 9
    # combined config: alpha = r and use_rslora=False => scaling exactly 1
    assert solve(mode="resolve_scaling", alpha=8, r=8) == 1.0
    # rslora would break the exact-1 invariant (why the config resets it)
    assert abs(solve(mode="resolve_scaling", alpha=8, r=8, use_rslora=True) - 8 / math.sqrt(8)) < 1e-12


def check_cat(solve):
    torch.manual_seed(0)
    A1 = torch.randn(3, 10)
    B1 = torch.randn(6, 3)
    A2 = torch.randn(2, 10)
    B2 = torch.randn(6, 2)
    out = solve(mode="merge_cat", A_list=[A1, A2], B_list=[B1, B2],
                weights=[0.5, -2.0], scalings=[2.0, 0.5])
    # A is scaled by weight * scaling and concatenated along dim 0 (rank)
    assert out["A"].shape == (5, 10)
    assert torch.equal(out["A"][:3], A1 * 0.5 * 2.0)
    assert torch.equal(out["A"][3:], A2 * -2.0 * 0.5)
    # B is unscaled and concatenated along dim 1
    assert out["B"].shape == (6, 5)
    assert torch.equal(out["B"][:, :3], B1)
    assert torch.equal(out["B"][:, 3:], B2)
    # empty source list raises
    try:
        solve(mode="merge_cat", A_list=[], B_list=[], weights=[], scalings=[])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    # anchor
    assert abs(out["A"][0, 0].item() - (-2.3104118002341765)) < 1e-9, out["A"][0, 0].item()


def check_linear(solve):
    torch.manual_seed(1)
    A1 = torch.randn(4, 10)
    B1 = torch.randn(6, 4)
    A2 = torch.randn(4, 10)
    B2 = torch.randn(6, 4)
    weights, scalings = [0.5, -2.0], [2.0, 1.0]
    out = solve(mode="merge_linear", A_list=[A1, A2], B_list=[B1, B2],
                weights=weights, scalings=scalings)
    # sqrt-split: sign on the A side only; B side always positive
    wa = [math.sqrt(abs(w * s)) * (1 if w * s >= 0 else -1) for w, s in zip(weights, scalings)]
    wb = [math.sqrt(abs(w * s)) for w, s in zip(weights, scalings)]
    assert torch.allclose(out["A"], wa[0] * A1 + wa[1] * A2, atol=1e-12, rtol=0)
    assert torch.allclose(out["B"], wb[0] * B1 + wb[1] * B2, atol=1e-12, rtol=0)
    assert wa == [1.0, -math.sqrt(2.0)] and wb == [1.0, math.sqrt(2.0)]
    # rough approximation: cross terms mean A@B != sum(w*s*B@A) for two sources
    want = weights[0] * scalings[0] * (B1 @ A1) + weights[1] * scalings[1] * (B2 @ A2)
    assert not torch.allclose(out["B"] @ out["A"], want, atol=1e-6)
    # but it is exact for a single source
    out1 = solve(mode="merge_linear", A_list=[A1], B_list=[B1], weights=[0.5], scalings=[2.0])
    assert torch.allclose(out1["B"] @ out1["A"], 1.0 * (B1 @ A1), atol=1e-12, rtol=0)
    # anchor
    assert abs(out["A"][0, 0].item() - (-0.005048035905528103)) < 1e-9, out["A"][0, 0].item()


def check_svd(solve):
    torch.manual_seed(2)
    A1 = torch.randn(3, 10) * 0.1
    B1 = torch.randn(6, 3) * 0.1
    A2 = torch.randn(3, 10) * 0.1
    B2 = torch.randn(6, 3) * 0.1
    weights, scalings = [0.5, -2.0], [2.0, 0.5]
    # full rank: reconstruction is exact and equals the scaled weighted sum of deltas
    out = solve(mode="merge_svd", A_list=[A1, A2], B_list=[B1, B2],
                weights=weights, scalings=scalings, new_rank=6, full_matrices=False)
    want = weights[0] * scalings[0] * (B1 @ A1) + weights[1] * scalings[1] * (B2 @ A2)
    assert out["A"].shape == (6, 10) and out["B"].shape == (6, 6)
    assert torch.allclose(out["B"] @ out["A"], want, atol=1e-10, rtol=0)
    # kohya convention: singular values are absorbed into B, not split as sqrt
    svals = torch.linalg.svdvals(want)[:6]
    assert torch.allclose(out["B"].pow(2).sum(0).sqrt(), svals, atol=1e-9, rtol=0)
    # svd branch uses RAW weights on already-scaled deltas (sign handled naturally)
    d1 = (B1 @ A1) * scalings[0]
    d2 = (B2 @ A2) * scalings[1]
    assert torch.allclose(out["B"] @ out["A"], 0.5 * d1 - 2.0 * d2, atol=1e-10, rtol=0)
    # rank truncation: best rank-r approximation of the combined delta
    out_r = solve(mode="merge_svd", A_list=[A1, A2], B_list=[B1, B2],
                  weights=weights, scalings=scalings, new_rank=2, full_matrices=False)
    U, S, Vh = torch.linalg.svd(want, full_matrices=False)
    best = (U[:, :2] * S[:2]) @ Vh[:2]
    assert torch.allclose(out_r["B"] @ out_r["A"], best, atol=1e-10, rtol=0)
    # svd_clamp: quantile of the concatenated U/Vh values, symmetric clamping
    out_raw = solve(mode="merge_svd", A_list=[A1], B_list=[B1],
                    weights=[1.0], scalings=[1.0], new_rank=3, full_matrices=False)
    out_c = solve(mode="merge_svd", A_list=[A1], B_list=[B1],
                  weights=[1.0], scalings=[1.0], new_rank=3, clamp=0.9, full_matrices=False)
    hi = torch.quantile(torch.cat([out_raw["B"].flatten(), out_raw["A"].flatten()]), 0.9)
    assert hi > 0  # upper quantile must be positive or the symmetric clamp inverts
    assert torch.allclose(out_c["B"], out_raw["B"].clamp(-hi, hi), atol=1e-12, rtol=0)
    assert torch.allclose(out_c["A"], out_raw["A"].clamp(-hi, hi), atol=1e-12, rtol=0)
    assert (out_c["A"] != out_raw["A"]).any()  # the clamp actually binds on this data
    # empty source list raises
    try:
        solve(mode="merge_svd", A_list=[], B_list=[], weights=[], scalings=[], new_rank=1)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    # anchor (reconstruction entry, sign-invariant)
    assert abs((out["B"] @ out["A"])[0, 0].item() - (-0.045955666016711375)) < 1e-9, (out["B"] @ out["A"])[0, 0].item()


def check_bias(solve):
    torch.manual_seed(3)
    b1 = torch.randn(6)
    b2 = torch.randn(6)
    # only sources with lora_bias contribute, scaled by weight * scaling
    out = solve(mode="merge_bias", biases=[b1, b2], weights=[0.5, 3.0], scalings=[2.0, 0.5])
    assert torch.allclose(out["bias"], 1.0 * b1 + 1.5 * b2, atol=1e-12, rtol=0)
    out2 = solve(mode="merge_bias", biases=[b1, None], weights=[0.5, 3.0], scalings=[2.0, 0.5])
    assert torch.allclose(out2["bias"], b1, atol=1e-12, rtol=0)
    # all-None raises
    try:
        solve(mode="merge_bias", biases=[None, None], weights=[1.0, 1.0], scalings=[1.0, 1.0])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    # anchor
    assert abs(out["bias"][0].item() - 1.3095187258920458) < 1e-9, out["bias"][0].item()


CHECKS = [
    "check_resolve",
    "check_cat",
    "check_linear",
    "check_svd",
    "check_bias",
]


def check(solve):
    for name in CHECKS:
        globals()[name](solve)
