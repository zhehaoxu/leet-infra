"""PEFT: EVA explained-variance rank allocation (official ml-jku/EVA src/svd.py).

EVA (Explained Variance Adaptation, arXiv:2410.07170) initializes LoRA adapters
in a data-driven manner and adaptively allocates ranks across layers according
to the variance each layer's principal components explain.

Ledger (official ml-jku/EVA implementation, svd_precompute.py -> src/svd.py):

- metric (4 modes): raw = explained_variance_; ratio = ev/ev.sum();
  sum = ev/ev.sum(); max = ev/ev.max()
- rank distribution: per-layer metric vector truncated to round(rank*rho)
  components; ALL (layer, value) pairs across layers are stacked and argsorted
  descending; the top `len(layers) * rank` entries (the global rank budget)
  vote per layer via Counter (layers with no vote get 0); an equal-inputs
  correction then ensures the SVD source layer keeps a rank >= its
  equivalent layer (single-direction swap, skipped when already >=)
- final assembly: per layer u = components[:rank_dist]; `whiten` divides by
  sqrt(singular_values), else `scale_by_singular_values` multiplies by
  s/s.max() (whiten wins when both are set — official elif)
"""

import torch


def _get_metric(ev, metric):
    if metric == "raw":
        return ev
    elif metric == "ratio":
        return ev / ev.sum()
    elif metric == "sum":
        return ev / ev.sum()
    elif metric == "max":
        return ev / ev.max()
    else:
        raise ValueError(f"Invalid metric: {metric}")


def _get_rank_distribution(exp_vars, hook_layer_map, equal_inputs_map, metric, rank_budget, max_components):
    from collections import Counter

    ev = {k: _get_metric(v, metric)[:max_components] for k, v in exp_vars.items()}
    keys, values = zip(*[(k, c) for k, name in hook_layer_map.items() for c in ev[name]])
    idx = torch.stack(values).argsort(descending=True)
    counts = Counter([keys[i] for i in idx[:rank_budget]])
    counts = {k: counts.get(k, 0) for k in hook_layer_map.keys()}  # add layers with 0 rank
    for k, k_hook in equal_inputs_map.items():
        # ensure hook layers have the highest rank if they are equal to another layer
        rank, rank_hook = counts[k], counts[k_hook]
        if rank_hook >= rank:
            continue
        counts[k_hook], counts[k] = rank, rank_hook
    return counts


def compute_rank_distribution(exp_vars, hook_layer_map, equal_inputs_map, rank, rho, metric):
    """Public wrapper mirroring compute_svd's bookkeeping around _get_rank_distribution."""
    assert rho >= 1, "early_stop_rho must be >= 1"
    max_components = round(rank * rho)
    rank_budget = len(hook_layer_map) * rank
    return _get_rank_distribution(exp_vars, hook_layer_map, equal_inputs_map, metric, rank_budget, max_components)


def scale_components(u, s, whiten=False, scale_by_singular_values=False):
    """Official assembly scaling: whiten (elif) scale_by_singular_values."""
    if whiten:
        return u / s.sqrt().reshape(-1, 1)
    elif scale_by_singular_values:
        s = s / s.max()
        return u * s.reshape(-1, 1)
    return u


def assemble_svd_dict(components, singular_values, hook_layer_map, rank_dist, whiten=False, scale_by_singular_values=False):
    """Official svd_dict assembly: skip rank-0 layers, slice components[:rank], scale."""
    svd_dict = {}
    for name, rank in rank_dist.items():
        if rank == 0:
            continue
        src = hook_layer_map[name]
        u = components[src][:rank]
        svd_dict[name] = scale_components(u, singular_values[src][:rank], whiten, scale_by_singular_values)
    return svd_dict


def solve(mode=None, **kw):
    if mode == "metric":
        return _get_metric(kw["explained_variance"], kw["metric"])
    if mode == "rank_dist":
        return compute_rank_distribution(
            kw["exp_vars"], kw["hook_layer_map"], kw.get("equal_inputs_map", {}),
            kw["rank"], kw["rho"], kw["metric"],
        )
    if mode == "scale_components":
        return scale_components(
            kw["components"], kw["singular_values"],
            kw.get("whiten", False), kw.get("scale_by_singular_values", False),
        )
    if mode == "svd_dict":
        return assemble_svd_dict(
            kw["components"], kw["singular_values"], kw["hook_layer_map"], kw["rank_dist"],
            kw.get("whiten", False), kw.get("scale_by_singular_values", False),
        )
    raise ValueError(f"unknown mode {mode!r}")
