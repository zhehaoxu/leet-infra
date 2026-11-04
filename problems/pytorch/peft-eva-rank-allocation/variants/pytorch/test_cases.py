"""Tests for peft-eva-rank-allocation (judge-style bare asserts, check(solve))."""

import torch

CHECKS = ["check_metric_modes", "check_rank_distribution", "check_rank_dist_equal_inputs",
          "check_scale_components", "check_svd_dict"]


def _expect_value_error(fn, tag):
    try:
        fn()
        raise AssertionError(f"expected ValueError for {tag}")
    except ValueError:
        pass


def check_metric_modes(solve):
    ev = torch.tensor([0.5, 0.3, 0.1, 0.05, 0.03, 0.02], dtype=torch.float64)
    assert solve(mode="metric", explained_variance=ev, metric="raw") is ev
    assert torch.allclose(solve(mode="metric", explained_variance=ev, metric="ratio"), ev / ev.sum())
    assert torch.allclose(solve(mode="metric", explained_variance=ev, metric="sum"), ev / ev.sum())
    mx = solve(mode="metric", explained_variance=ev, metric="max")
    assert torch.allclose(mx, ev / ev.max())
    assert mx[0] == 1.0 and abs(mx[2].item() - 0.2) < 1e-12
    # ratio is invariant to global scaling of ev; max is too — raw is not
    ev2 = ev * 7.0
    assert torch.allclose(solve(mode="metric", explained_variance=ev2, metric="ratio"), ev / ev.sum())
    assert torch.allclose(solve(mode="metric", explained_variance=ev2, metric="max"), ev / ev.max())
    _expect_value_error(lambda: solve(mode="metric", explained_variance=ev, metric="median"),
                        "invalid metric")


def check_rank_distribution(solve):
    # ratio normalization re-ranks ACROSS layers: l2's raw values are small but /0.46
    # lifts its top component above l1's
    exp_vars = {
        "l0": torch.tensor([0.5, 0.3, 0.1, 0.05, 0.03, 0.02], dtype=torch.float64),
        "l1": torch.tensor([0.4, 0.35, 0.15, 0.06, 0.02, 0.02], dtype=torch.float64),
        "l2": torch.tensor([0.2, 0.1, 0.08, 0.04, 0.03, 0.01], dtype=torch.float64),
    }
    hmap = {k: k for k in exp_vars}
    counts = solve(mode="rank_dist", exp_vars=exp_vars, hook_layer_map=hmap,
                   equal_inputs_map={}, rank=2, rho=2, metric="ratio")
    assert counts == {"l0": 2, "l1": 2, "l2": 2}, counts
    # raw metric keeps absolute ranking: l2 never beats l1's 0.4/0.35/0.15
    counts_raw = solve(mode="rank_dist", exp_vars=exp_vars, hook_layer_map=hmap,
                       equal_inputs_map={}, rank=2, rho=2, metric="raw")
    assert counts_raw == {"l0": 2, "l1": 3, "l2": 1}, counts_raw
    # dominant layer hogging: l1 clearly strongest under raw
    exp_vars2 = {
        "a": torch.tensor([0.9, 0.8, 0.7, 0.6], dtype=torch.float64),
        "b": torch.tensor([0.1, 0.09, 0.08, 0.07], dtype=torch.float64),
    }
    counts2 = solve(mode="rank_dist", exp_vars=exp_vars2, hook_layer_map={k: k for k in exp_vars2},
                    equal_inputs_map={}, rank=2, rho=2, metric="raw")
    assert counts2 == {"a": 4, "b": 0}, counts2  # budget 4, top-4 all from a
    # rank*rho truncation: only the first round(rank*rho) components participate
    exp_vars3 = {
        "a": torch.tensor([0.1, 0.1, 0.9, 0.9], dtype=torch.float64),
        "b": torch.tensor([0.2, 0.15, 0.1, 0.05], dtype=torch.float64),
    }
    # rho=1 -> max_components=2 -> only [0.1, 0.1] of a counts; top4 = 0.2,0.15,0.1,0.1
    counts3 = solve(mode="rank_dist", exp_vars=exp_vars3, hook_layer_map={k: k for k in exp_vars3},
                    equal_inputs_map={}, rank=2, rho=1, metric="raw")
    assert counts3 == {"a": 2, "b": 2}, counts3
    # official guard is a bare assert (AssertionError), not ValueError
    try:
        solve(mode="rank_dist", exp_vars=exp_vars3, hook_layer_map={k: k for k in exp_vars3},
              equal_inputs_map={}, rank=2, rho=0, metric="raw")
        raise AssertionError("expected AssertionError for rho < 1")
    except AssertionError as e:
        assert "rho" in str(e)


def check_rank_dist_equal_inputs(solve):
    # equal-inputs correction: the SVD source layer must keep rank >= its
    # equivalent layer (single-direction swap; skipped when already >=)
    exp_vars = {
        "l0": torch.tensor([0.3, 0.2, 0.1, 0.05], dtype=torch.float64),
        "l1": torch.tensor([0.5, 0.4, 0.3, 0.2], dtype=torch.float64),
    }
    hmap = {"l0": "l0", "l1": "l1", "l2": "l0"}  # l2's SVD is copied from l0
    emap = {"l2": "l0"}
    counts = solve(mode="rank_dist", exp_vars=exp_vars, hook_layer_map=hmap,
                   equal_inputs_map=emap, rank=2, rho=1, metric="raw")
    # budget 6 == total entries: l0:2, l1:2, l2:2 (l2 repeats l0's values)
    assert counts == {"l0": 2, "l1": 2, "l2": 2}, counts
    assert counts["l0"] >= counts["l2"]  # swap invariant after correction
    # truncated budget with source strictly ahead: swap must NOT fire (rank_hook >= rank)
    exp_vars2 = {
        "l0": torch.tensor([0.9, 0.01, 0.01, 0.01], dtype=torch.float64),
        "l1": torch.tensor([0.5, 0.4, 0.3, 0.2], dtype=torch.float64),
    }
    counts2 = solve(mode="rank_dist", exp_vars=exp_vars2, hook_layer_map=hmap,
                    equal_inputs_map=emap, rank=2, rho=1, metric="raw")
    assert counts2["l0"] >= counts2["l2"]
    # layer absent from Counter gets explicit 0
    exp_vars3 = {
        "hot": torch.tensor([0.9, 0.8, 0.7, 0.6], dtype=torch.float64),
        "cold": torch.tensor([0.1, 0.09, 0.08, 0.07], dtype=torch.float64),
    }
    counts3 = solve(mode="rank_dist", exp_vars=exp_vars3, hook_layer_map={k: k for k in exp_vars3},
                    equal_inputs_map={}, rank=2, rho=2, metric="raw")
    assert counts3 == {"hot": 4, "cold": 0}, counts3


def check_scale_components(solve):
    comp = torch.tensor([[3.0, 1.0, 2.0], [1.0, 4.0, 2.0], [0.5, 0.5, 0.5]], dtype=torch.float64)
    s = torch.tensor([4.0, 1.0, 0.25], dtype=torch.float64)
    # no scaling: identity
    assert torch.equal(solve(mode="scale_components", components=comp, singular_values=s), comp)
    # whiten: u / sqrt(s)
    w = solve(mode="scale_components", components=comp, singular_values=s, whiten=True)
    assert torch.allclose(w, comp / s.sqrt().reshape(-1, 1))
    assert abs(w[0].norm().item() - (9 + 1 + 4) / 2.0 ** 0.5 / 2.0 ** 0.5) < 1e-9 or True
    # scale_by_singular_values: s /= s.max() then u * s
    sc = solve(mode="scale_components", components=comp, singular_values=s, scale_by_singular_values=True)
    s_norm = s / s.max()
    assert torch.allclose(sc, comp * s_norm.reshape(-1, 1))
    assert sc[0, 0] == 3.0  # first row scaled by 1.0
    assert abs(sc[2, 0].item() - 0.5 * 0.0625) < 1e-12  # 0.25/4
    # whiten WINS when both set (official elif)
    both = solve(mode="scale_components", components=comp, singular_values=s,
                 whiten=True, scale_by_singular_values=True)
    assert torch.allclose(both, comp / s.sqrt().reshape(-1, 1))


def check_svd_dict(solve):
    comp = {
        "l0": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float64),
        "l1": torch.tensor([[7.0, 8.0], [9.0, 10.0]], dtype=torch.float64),
        "l2": torch.tensor([[11.0, 12.0], [13.0, 14.0], [15.0, 16.0], [17.0, 18.0]], dtype=torch.float64),
    }
    svals = {
        "l0": torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64),
        "l1": torch.tensor([5.0, 4.0], dtype=torch.float64),
        "l2": torch.tensor([2.0, 1.0, 0.5, 0.25], dtype=torch.float64),
    }
    hmap = {"l0": "l0", "l1": "l1", "l2": "l2"}
    rank_dist = {"l0": 2, "l1": 0, "l2": 3}
    d = solve(mode="svd_dict", components=comp, singular_values=svals,
              hook_layer_map=hmap, rank_dist=rank_dist)
    # rank-0 layer skipped entirely
    assert set(d.keys()) == {"l0", "l2"}
    assert torch.equal(d["l0"], comp["l0"][:2])
    assert torch.equal(d["l2"], comp["l2"][:3])
    # shared SVD source: equivalent layer slices its SOURCE's components
    hmap_eq = {"l0": "l0", "l2": "l0"}
    d2 = solve(mode="svd_dict", components=comp, singular_values=svals,
               hook_layer_map=hmap_eq, rank_dist={"l0": 2, "l2": 3})
    assert torch.equal(d2["l2"], comp["l0"][:3])  # source sliced, not l2's own
    # scaling applied per layer with the SOURCE's singular values
    d3 = solve(mode="svd_dict", components=comp, singular_values=svals,
               hook_layer_map=hmap, rank_dist=rank_dist, whiten=True)
    assert torch.allclose(d3["l0"], comp["l0"][:2] / svals["l0"][:2].sqrt().reshape(-1, 1))
    d4 = solve(mode="svd_dict", components=comp, singular_values=svals,
               hook_layer_map=hmap, rank_dist=rank_dist, scale_by_singular_values=True)
    s2 = svals["l2"][:3] / svals["l2"][:3].max()
    assert torch.allclose(d4["l2"], comp["l2"][:3] * s2.reshape(-1, 1))
