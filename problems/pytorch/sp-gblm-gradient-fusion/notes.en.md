# Notes: GBLM Gradient-Fused Saliency Rowwise Pruning

## Upstream source

Paper: *GBLM-Pruner: Gradient-based Language Model Pruner* (2311.04902, ICLR 2024). The repo originally listed as `IST-DASLab/GBLM-Pruner` has moved to **`VILA-Lab/GBLM-Pruner`** (fetched 2026-10-31, `main` branch). Line-by-line alignment against:

- `lib/prune.py::prune_gblm` (L216–311) — the core fusion formula and mask generation;
- `lib/prune.py::prune_wanda` (L314–401) — baseline; structurally identical minus the gradient term;
- `gradient_computation.py::gradient_computation` (L119–157) — two-stage gradient aggregation;
- `lib/prune.py::return_given_alpha` (L151–157) — the `use_variant` bisection branch (**not** reproduced here, see below).

## Line-level semantics

1. **Gradient aggregation (gradient_computation.py L155–156, L207–209)**: per sample, `g·scale` is accumulated into two buffers:
   - `gradients_l1 += |g·scale|` (fp16);
   - `gradients_l2 += (g·scale)²` (fp32), with a single `sqrt` applied before saving.
   `prune_gblm` loads the **L2 file** (`main.py --prune_method gblm` defaults to `gradients_aggregrate_norm_l2_*`), hence `G = sqrt(Σ_s (g_s·scale)²)`. An L1 file exists but the main path never uses it.
2. **Fusion (prune.py L259–267)**:
   ```python
   W_metric = |W| * torch.sqrt(scaler_row.reshape((1,-1)))     # L259, Wanda term
   W_metric_grad = |W| * |G|                                    # L262
   W_metric = W_metric + W_metric_grad                          # L263, non-inv
   # inv branch L267: W_metric = W_metric * (1/(|G| + 1e-8))
   ```
   The gradient term carries **no** `√s_row` factor — it multiplies `|W|` directly. This is additive fusion of two evidence terms ("column activation RMS" + "elementwise gradient"), not `|W|·(√s + G)` with G rescaled by `√s`. `small_value = 1e-8` appears only in the inv branch.
3. **scaler_row (layerwrapper.py, same family as SparseGPT/Wanda)**: `scaler_row += x²` accumulated over batches (sum, not divided by sample count). This problem keeps problem 581's **mean convention**, declared in the description — `sqrt(sum) = sqrt(mean·n_tokens)` with `n_tokens` a per-row constant factor applied to whole columns, so **masks are unchanged** and only `metric` values rescale; judging compares under the mean convention.
4. **Mask (prune.py L277–301)**: `torch.sort(W_metric, dim=-1, stable=True)`, then `sort_res[1][:, :int(cols·sparsity)]` — **equal per-row truncation** (Wanda-style), not a global threshold. `stable=True` resolves ties toward smaller indices.
5. **2:4 branch (L272–275)**: identical to Wanda's (top-n smallest per m-block), applied to the fused metric.
6. **`use_variant` bisection (L279–297)**: per-row cumulative sums with an alpha binary search toward the target sparsity (the "Wanda variant"). Involves a convergence loop and float comparisons — **not reproduced**; `prune_gradient` (L182–213, gradient-only) and `prune_magnitude` (L159–180) are baselines only.

## Difference from problem 581 (dedup statement)

| Aspect | 581 Wanda | This problem (GBLM) |
| --- | --- | --- |
| Saliency | `\|W\|·√mean(x²)` | that ⊕ `\|W\|·sqrt(Σ(g·scale)²)` |
| Gradient input | none | per-sample gradients `(S, out, in)` + `scale` |
| Variant | none | `gradient_inv` (`base/(G+1e-8)`) |
| Focus | activation RMS × rowwise truncation | fusion weight ratio, L2 vs L1 aggregation, scale sensitivity, inv flip |

The mask skeleton (stable sort, per-row int truncation, N:M windows) is shared with 581, but the graded core is the four fusion traps: aggregation mode, missing `scale`, inv denominator eps, and the base the inv variant divides. Mutation testing: 7/7 caught (grad-l1-agg / no-scale / grad-term-missing / inv-no-eps / inv-on-wrong-base / sort-descending / sum-scaler).

## Deliberate deviations from upstream

- **fp16 → float64**: upstream mixes fp16/fp32 accumulation; judging uses uniform float64 — semantically equivalent (elementwise fusion, no order sensitivity).
- **scaler_row mean convention**: see item 3; masks unaffected.
- **`use_variant` alpha bisection, `prune_gradient`, `prune_magnitude` not reproduced**: not GBLM's core contribution and/or contain convergence loops.
- **No data pipeline**: upstream runs wikitext/c4 forwards for activations and backwards for gradients; judging passes `activations` and `grad_samples` tensors directly.
- **`scale` made an explicit parameter**: upstream CLI default is 100; test cases give it explicitly (including 1 and 10 to expose the weight-ratio effect). Semantics unchanged.

## Test design

- Hand-computed 2×4 case: zero `√s_row` column, zero gradient column, stable tie-break;
- Scale-sensitivity case: same inputs, scale 1 vs 10 → different masks (proves `scale` is not a droppable parameter);
- inv-flip case: non-inv and inv prune different columns; zero-gradient slot yields metric 1e9 and survives while 8/20 = 0.4 is pruned first;
- 2:4 window case: metric ties `[2,2]` under stable tie-break;
- Multi-sample L2 aggregation + random-shape comparison against an independent numpy oracle (rtol 1e-9);
- 11 validation cases: 3-D weight, column mismatch, dtype mismatch, grad shape/sample count, sparsity range, non-positive scale, non-bool inv, and the three N:M constraints.
