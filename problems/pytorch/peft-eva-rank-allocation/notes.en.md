# Notes: EVA's Explained-Variance-Driven Global Rank Allocation

## Source

EVA (One Initialization to Rule them All: Fine-tuning via Explained Variance Adaptation, arXiv:2410.07170, Paischer et al., ELLIS Unit LIT AI Lab / NXAI): run **incremental PCA** (torch_incremental_pca.IncrementalPCA's partial_fit) on each LoRA target module's input activations, and schedule LoRA's rank budget across layers by "explained variance"; the initialization uses each layer's principal directions as A (optionally whitened / scaled by singular values). Official repo ml-jku/EVA: `svd_precompute.py` (CLI shell) -> `src/svd.py::compute_svd` (235-line core) + SVDHook (incremental PCA + cosine-similarity convergence) + HashHook (equal-input detection). The PEFT side ships via a PR (init option, not a standalone tuner directory).

## Line-by-line translation of the official implementation

**Metric four modes (`_get_metric`)**:

```python
if metric == "raw":     return svd.explained_variance_
elif metric == "ratio": return svd.explained_variance_ratio_   # == ev/ev.sum()
elif metric == "sum":   return svd.explained_variance_ / svd.explained_variance_.sum()
elif metric == "max":   return svd.explained_variance_ / svd.explained_variance_.max()
else: raise ValueError
# note: ratio and sum are equivalent official branches, both kept in the judging domain
```

**Global rank allocation (`_get_rank_distribution`)**:

```python
exp_vars = {k: _get_metric(h.svd, metric)[:max_components] for k, h in hooks.items()}
#   max_components = round(rank * rho)   # rho >= 1 guarded by an assert
keys, values = zip(*[(k, c) for k, name in hook_layer_map.items() for c in exp_vars[name]])
#   hook_layer_map: {layer: svd_source} — an equivalent layer expands its SOURCE's vector (duplicated values!)
idx = torch.stack(values).argsort(descending=True)
counts = Counter([keys[i] for i in idx[:rank_budget]])       # rank_budget = len(hooks)*rank
counts = {k: counts.get(k, 0) for k in hook_layer_map.keys()} # explicit 0 for voteless layers
for k, k_hook in equal_inputs_map.items():
    rank, rank_hook = counts[k], counts[k_hook]
    if rank_hook >= rank:
        continue                                              # one-direction swap
    counts[k_hook], counts[k] = rank, rank_hook
```

**Assembly (`compute_svd` tail)**:

```python
for name, rank in rank_dist.items():
    if rank == 0: continue                                    # rank-0 layers excluded
    hook = hooks[layer_hook_map[name]]                        # equivalent layers take the SOURCE's hook
    u = hook.svd.components_[:rank]
    if whiten:                    u /= hook.svd.singular_values_[:rank].sqrt().reshape(-1, 1)
    elif scale_by_singular_values:                             # elif: whiten wins when both set
        s = hook.svd.singular_values_[:rank]
        s /= s.max()
        u *= s.reshape(-1, 1)
svd_dict[name] = u
```

**SVDHook convergence (excluded)**: after each batch's `partial_fit`, `cosine_similarity(components, previous) >= sim_thresh` per-component bool; the loop exits only when all converge (and `i > min_batches`). The judging domain takes explained_variance explicitly, so the whole loop is excluded.

## Exclusions and red lines in the judging domain

- IncrementalPCA / partial_fit / real-model forward hooks / DataLoader / attention_mask & labels masks: real-forward red line.
- The convergence loop (sim_thresh / min_batches / convergence_dict): data-flow dependent, excluded.
- HashHook's collision detection (`hash(tuple(tensor.view(-1).tolist()))`): Python hash randomization is unjudgeable — the judging domain takes equal_inputs_map explicitly.
- `torch.save(svd_dict)` serialization and fileref name assembly: IO red line.

## Key quirks (verified line-by-line against the source)

1. **ratio and sum are two mathematically equivalent official branches** (both ev/ev.sum()) — the judging domain keeps both, and the tests verify the equivalence.
2. **Per-layer normalization before cross-layer ranking**: normalization lets a low-variance layer overtake (in the example l2 has the smallest absolute values yet takes the full budget share) — raw and ratio can allocate completely differently.
3. **An equivalent layer expands its source's vector** (duplicated values): argsort orders equal values by index (deterministic CPU implementation), and the source layer appears earlier in hook_layer_map -> the source always enters the budget first -> the `if rank_hook >= rank: continue` swap branch **never fires** in the judging domain (truly equivalent), but the official logic is kept in full and the tests only assert the post-swap invariant `counts[source] >= counts[equivalent]`.
4. **whiten/scale elif**: under an if-return structure the elif and two independent ifs behave identically (the first branch always returns) — "whiten wins when both are set" is official semantics rather than a conventional else-if exclusivity; the tests pin it with a both-True case.

## Mutation testing (23/23, 2 equivalent exclusions)

- Metric: ratio drops normalization / sum swaps to max / max swaps to sum / silent invalid metric — all CAUGHT.
- Allocation: truncation +1 / max_components -1 / ascending argsort / budget -1 / voteless default 1 / expansion drops the first component / keys use the source name / budget formula loses rank / swap self-assignment — all CAUGHT except swap.
- Scaling: whiten without sqrt / max -> min / multiply -> divide / reshape axis flip / identity zeroed — all CAUGHT.
- Assembly: skip disabled / slice +1 / source looked up as itself / singular-value slice offset / key written as source / inverted skip — all CAUGHT.
- **2 truly equivalent exclusions**: (1) m-rd-swap (swap self-assignment) — in the judging domain an equivalent layer's expanded values equal its source's, and argsort's index-order equal values mean the source always enters the budget first, so the swap branch never fires (quirk 3); (2) m-sc-elif (elif -> if) — under if-return the first branch always returns, so both spellings are identical (quirk 4).

## Boundary with existing problems

- The sparsity SVD problems (svdllm/zssvd etc.): those apply SVD to **weight whitening/rank reduction** with spectra from weight matrices; this problem's input is **activation explained_variance vectors** (explicit inputs) and tests the **cross-layer budget allocation ledger**, performing no SVD at all.
- `peft-tinylora-svd-projection` (batch 795): frozen SVD factor parameterization, no cross-layer scheduling.
- PSOFT/OSF (group 1 candidates, not filed): top-k SVD mask initialization needing real-SVD oracle agreement; this problem has no SVD dependency (the official svd precompute is offline; the judging domain takes it as input).
