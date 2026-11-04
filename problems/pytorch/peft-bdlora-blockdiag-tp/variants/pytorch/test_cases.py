"""BD-LoRA test cases. Judging convention: string CHECKS + check(solve)."""
import math

import torch

torch.set_default_dtype(torch.float64)


def check_bd_forward(solve):
    torch.manual_seed(0)
    B, inp, out_f, nb = 4, 8, 6, 2
    x = torch.randn(B, inp)
    w = torch.randn(out_f, inp // nb)
    y = solve(mode="bd_forward", x=x, weight=w, nblocks=nb)
    assert y.shape == (B, out_f), y.shape
    # per-shard account: block i maps x[:, i*m:(i+1)*m] with w[i*n:(i+1)*n, :]
    m, n = inp // nb, out_f // nb
    y2 = torch.cat([x[:, i*m:(i+1)*m] @ w[i*n:(i+1)*n, :].t() for i in range(nb)], dim=-1)
    assert torch.allclose(y, y2, atol=1e-9), "per-shard account mismatch"
    # anchors
    assert abs(y[0, 0].item() - (-1.2027563104668666)) < 1e-9, y[0, 0].item()
    assert abs(y[-1, -1].item() - 2.3503342488855252) < 1e-9, y[-1, -1].item()
    # 3-D input keeps the leading dims
    x3 = torch.randn(2, 5, inp)
    y3 = solve(mode="bd_forward", x=x3, weight=w, nblocks=nb)
    assert y3.shape == (2, 5, out_f), y3.shape
    # equivalence with the expanded block-diagonal matrix
    bd = solve(mode="block_diag_matrix", weight=w, nblocks=nb)
    assert torch.allclose(y, x @ bd.t(), atol=1e-9), "einsum vs block_diag mismatch"
    # divisibility validation
    try:
        solve(mode="bd_forward", x=torch.randn(2, 7), weight=torch.randn(6, 4), nblocks=2)
        raise AssertionError("expected ValueError for in_features not divisible")
    except ValueError:
        pass
    # a different nblocks re-shards the computation
    y1 = solve(mode="bd_forward", x=x, weight=torch.randn(8, 1), nblocks=8)
    assert y1.shape == (B, 8)


def check_block_diag(solve):
    torch.manual_seed(1)
    out_f, inp, nb = 6, 8, 2
    w = torch.randn(out_f, inp // nb)
    bd = solve(mode="block_diag_matrix", weight=w, nblocks=nb)
    assert bd.shape == (out_f, inp), bd.shape
    n, m = out_f // nb, inp // nb
    # off-diagonal blocks are exactly zero, diagonal blocks equal the shards
    for i in range(nb):
        for j in range(nb):
            blk = bd[i*n:(i+1)*n, j*m:(j+1)*m]
            if i == j:
                assert torch.equal(blk, w[i*n:(i+1)*n, :]), "diagonal block mismatch"
            else:
                assert blk.abs().sum().item() == 0.0, "off-diagonal block not zero"
    # anchors
    assert abs(bd[0, 0].item() - (-0.31128955419003207)) < 1e-12, bd[0, 0].item()
    # non-square blocks are supported: A factor (r, in) with r != in
    r = 4
    wa = torch.randn(r, inp // nb)
    bd_a = solve(mode="block_diag_matrix", weight=wa, nblocks=nb)
    assert bd_a.shape == (r, inp), bd_a.shape


def check_classify(solve):
    pa, pb = ["o_proj"], ["q_proj", "up_proj"]
    assert solve(mode="classify_target", target_name="o_proj",
                 patterns_a=pa, patterns_b=pb, match_strict=True) == "a"
    assert solve(mode="classify_target", target_name="up_proj",
                 patterns_a=pa, patterns_b=pb, match_strict=True) == "b"
    # match_strict=False lets unmatched targets fall back to plain LoRA
    assert solve(mode="classify_target", target_name="down_proj",
                 patterns_a=pa, patterns_b=pb, match_strict=False) == "plain"
    # matching both patterns raises
    try:
        solve(mode="classify_target", target_name="proj",
              patterns_a=["proj"], patterns_b=["proj"], match_strict=True)
        raise AssertionError("expected ValueError for both patterns")
    except ValueError:
        pass
    # match_strict with no match raises
    try:
        solve(mode="classify_target", target_name="down_proj",
              patterns_a=pa, patterns_b=pb, match_strict=True)
        raise AssertionError("expected ValueError for strict miss")
    except ValueError:
        pass


def check_init(solve):
    inp, out_f, r, nb = 8, 6, 4, 2
    wa = solve(mode="init_factors", in_features=inp, out_features=out_f,
               r=r, nblocks=nb, role="a", seed=0)
    assert wa.shape == (r, inp // nb), wa.shape
    # official gate: A is kaiming-uniform with a=sqrt(5) (bound sqrt(6/((1+a^2)*fan_in)))
    bound = math.sqrt(6.0 / ((1 + 5) * (inp // nb)))
    assert wa.abs().max().item() <= bound + 1e-12, "kaiming bound violated"
    assert abs(wa[0, 0].item() - 0.470053001806553) < 1e-12, wa[0, 0].item()
    # same seed reproduces; different seed differs
    wa2 = solve(mode="init_factors", in_features=inp, out_features=out_f,
                r=r, nblocks=nb, role="a", seed=0)
    assert torch.equal(wa, wa2), "kaiming init not reproducible under the same seed"
    # B is exactly zero (delta W = 0 at start)
    wb = solve(mode="init_factors", in_features=inp, out_features=out_f,
               r=r, nblocks=nb, role="b")
    assert wb.shape == (out_f, r // nb), wb.shape
    assert wb.abs().sum().item() == 0.0, "lora_B must be zero-initialized"
    # divisibility validation
    try:
        solve(mode="init_factors", in_features=7, out_features=out_f,
              r=r, nblocks=nb, role="a")
        raise AssertionError("expected ValueError for in_features not divisible")
    except ValueError:
        pass
    try:
        solve(mode="init_factors", in_features=inp, out_features=out_f,
              r=r, nblocks=nb, role="c")
        raise AssertionError("expected ValueError for unknown role")
    except ValueError:
        pass


def check_delta_and_layer(solve):
    torch.manual_seed(2)
    inp, out_f, r, nb = 8, 6, 4, 2
    wa = solve(mode="init_factors", in_features=inp, out_features=out_f,
               r=r, nblocks=nb, role="a", seed=3)
    wb = torch.randn(out_f, r // nb)
    scaling = 2.0
    delta = solve(mode="delta_weight", weight_a=wa, weight_b=wb,
                  nblocks=nb, scaling=scaling)
    assert delta.shape == (out_f, inp), delta.shape
    # block structure: only diagonal blocks are nonzero
    n, m, rb = out_f // nb, inp // nb, r // nb
    for i in range(nb):
        want = wb[i*n:(i+1)*n, :] @ wa[i*rb:(i+1)*rb, :]
        assert torch.allclose(delta[i*n:(i+1)*n, i*m:(i+1)*m], want * scaling, atol=1e-9)
    off = delta.clone()
    for i in range(nb):
        off[i*n:(i+1)*n, i*m:(i+1)*m] = 0.0
    assert off.abs().sum().item() < 1e-12, "delta must stay block-diagonal"
    # anchors
    assert abs(delta[0, 0].item() - (-0.44661549644151793)) < 1e-9, delta[0, 0].item()
    # merged base weight reproduces the unmerged two-step forward exactly
    base_w = torch.randn(out_f, inp)
    base_b = torch.randn(out_f)
    x = torch.randn(4, inp)
    y_un = solve(mode="layer_forward", x=x, base_weight=base_w, base_bias=base_b,
                 weight_a=wa, weight_b=wb, nblocks=nb, scaling=scaling)
    y_m = torch.nn.functional.linear(x, base_w + delta, base_b)
    assert torch.allclose(y_un, y_m, atol=1e-9), "merged vs unmerged mismatch"
    # fan_in_fan_out transposes the delta (Conv1D-style weights)
    delta_t = solve(mode="delta_weight", weight_a=wa, weight_b=wb,
                    nblocks=nb, scaling=scaling, fan_in_fan_out=True)
    assert torch.allclose(delta_t, delta.t(), atol=1e-9)
    # scaling enters the adapter output linearly
    y_half = solve(mode="layer_forward", x=x, base_weight=base_w, base_bias=base_b,
                   weight_a=wa, weight_b=wb, nblocks=nb, scaling=scaling / 2)
    manual = torch.nn.functional.linear(x, base_w, base_b)
    za = solve(mode="bd_forward", x=x, weight=wa, nblocks=nb)
    zb = solve(mode="bd_forward", x=za, weight=wb, nblocks=nb)
    assert torch.allclose(y_half, manual + zb * (scaling / 2), atol=1e-9)


CHECKS = [
    "check_bd_forward",
    "check_block_diag",
    "check_classify",
    "check_init",
    "check_delta_and_layer",
]


def check(solve):
    for name in CHECKS:
        globals()[name](solve)
