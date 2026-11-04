# PEFT: EVA's Explained-Variance-Driven Global Rank Allocation

## Problem Description

**EVA** (Explained Variance Adaptation, arXiv:2410.07170, ELLIS Unit LIT AI Lab, now in a PEFT PR) schedules LoRA's rank budget **jointly across layers**: instead of a fixed rank per layer, it runs incremental PCA on each layer's inputs and awards the global budget of `len(layers) x rank` components to the "most deserving" layers by **explained variance** — high-variance layers get more rank, low-variance layers may get 0.

The deterministic ledger of the official ml-jku/EVA implementation (`src/svd.py::compute_svd`; all anchors re-computable offline):

```
# 1) metric (4 modes, _get_metric)
raw   = explained_variance_
ratio = ev / ev.sum()        # per-layer normalization!
sum   = ev / ev.sum()
max   = ev / ev.max()

# 2) global rank allocation (_get_rank_distribution)
ev = {layer: metric(layer_vector)[: round(rank * rho)]}   # first rank*rho components only
flatten ALL (layer, value) pairs -> argsort descending -> top len(layers)*rank (the
global budget) each cast one vote
counts = Counter(voted layers); layers with no vote get an explicit 0
# equal-inputs correction: if an equivalent layer's SVD source has a lower rank,
# swap (one-direction, skipped when the computed layer's rank is already >=)

# 3) assembly (svd_dict)
u = components[source][:rank_dist[layer]]                 # rank-0 layers skipped entirely
whiten: u /= sqrt(singular_values)                        # else (elif!)
scale_by_singular_values: s /= s.max(); u *= s            # whiten wins when both set
```

Three key conventions:

1. **Per-layer normalization BEFORE cross-layer ranking**: `ratio`/`sum`/`max` all normalize **within** a layer — a layer with small absolute values may overtake after normalization (e.g. ev=[0.2,0.1,0.08,0.04,0.03,0.01] sums to 0.46; its top component becomes 0.435, beating another layer's 0.35 second component). `raw` keeps absolute ordering.
2. **Truncation happens after normalization, before voting**: each layer contributes only its first `round(rank*rho)` components (`rho >= 1`, guarded by the official bare assert); the global budget `len(layers)*rank` truncates the flattened sequence.
3. **whiten and scale are mutually exclusive (elif)**: `whiten=True` divides by `sqrt(s)`; otherwise `scale_by_singular_values=True` multiplies by `s/max(s)`; when both are set whiten wins (official `elif`).

## Task

Implement four groups of entry points (the judging domain receives the `explained_variance` vectors pre-computed, explicitly):

- `mode="metric"`: a 1-D `explained_variance` tensor plus `metric` in {raw, ratio, sum, max} -> the normalized vector; an invalid metric raises `ValueError`.
- `mode="rank_dist"`: `exp_vars` (layer -> vector dict), `hook_layer_map` (layer -> SVD source layer dict), `equal_inputs_map` (equivalent layer -> source dict), `rank`, `rho`, `metric` -> a `{layer: rank}` dict; `rho < 1` raises `AssertionError` (official bare assert).
- `mode="scale_components"`: `components` (n, d), `singular_values` (n,), `whiten`, `scale_by_singular_values` -> the scaled (n, d); with both flags False, return as-is.
- `mode="svd_dict"`: `components`, `singular_values` (layer -> tensor dicts), `hook_layer_map`, `rank_dist` -> `{layer: u}`; rank-0 layers are excluded from the dict; equivalent layers slice their **source's** components and scale by the **source's** singular values.

## Example

Three layers, `rank=2, rho=2` (max_components=4, budget 6), metric=ratio:

```
l0 ev = [0.5, 0.3, 0.1, 0.05, 0.03, 0.02]   (sum=1.0)  -> [0.5, 0.3, 0.1, 0.05]
l1 ev = [0.4, 0.35, 0.15, 0.06, 0.02, 0.02] (sum=1.0)  -> [0.4, 0.35, 0.15, 0.06]
l2 ev = [0.2, 0.1, 0.08, 0.04, 0.03, 0.01]  (sum=0.46) -> [0.435, 0.217, 0.174, 0.087]
flattened descending top6: 0.5(l0), 0.435(l2), 0.4(l1), 0.35(l1), 0.3(l0), 0.217(l2)
-> {l0: 2, l1: 2, l2: 2}     # l2 has the smallest absolute values yet takes half the budget
```

`scale_components(u, s=[4,1,0.25], whiten=True, scale_by_singular_values=True)` -> `u / sqrt(s)` (whiten wins).

## Constraints

- Everything in float64; argsort via torch (descending); Counter's voteless layers get an explicit 0 (int).
- The `equal_inputs_map` swap fires only when the source layer's rank is **strictly lower** (equal ranks do not swap); in the judging domain the expanded values coincide so the swap is mostly a no-op, but implement it in full per the official code.
- The returned `svd_dict` contains only layers with rank > 0; each layer returns an independent tensor.
