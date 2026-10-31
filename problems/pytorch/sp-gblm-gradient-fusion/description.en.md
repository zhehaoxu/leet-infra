## Problem

Wanda (problem 581) scores weights by `|W|·RMS(activation column)`. GBLM-Pruner (ICLR 2024) argues pruning should also consult **training gradients** — a weight with a large gradient on fine-tuning samples is one the model still wants to move, so cutting it is costly.

This problem reproduces GBLM's fused saliency on given per-sample gradients and performs rowwise pruning. Inputs:

- `weight`: float matrix of shape `(out, in)`;
- `activations`: calibration activations of shape `(tokens, in)`;
- `grad_samples`: per-sample gradients of shape `(samples, out, in)`;
- `sparsity`: unstructured pruning ratio in `[0, 1]`;
- `scale`: gradient scaling factor (upstream default 100; given explicitly, must be positive);
- `gradient_inv`: whether to use the gradient-inverse variant;
- `prune_n` / `prune_m`: N:M parameters; `prune_n > 0` enables structured mode.

## Rules

**Step 1: activation column means.** `scaler_row = mean(activations², dim=tokens)`, a length-`in` vector; square-root it elementwise to get `√s_row`.

**Step 2: L2 gradient aggregation.** Scale each sample by `scale`, then sum squares over the sample dimension and take the square root:

```
G[i,j] = sqrt( Σ_s (grad_samples[s,i,j] · scale)² )
```

This mirrors upstream `gradient_computation.py`, which accumulates `Σ|g·scale|²` and takes one final `sqrt`; it is **not** a per-sample max and **not** the L1 aggregate `Σ|g|`.

**Step 3: fused saliency.** Let `base = |W| · √s_row` (the Wanda term, broadcast per column):

- non-inv: `metric = base + |W| · G` (two pieces of evidence added; the gradient term carries **no** `√s_row` factor);
- inv variant: `metric = base / (G + 1e-8)` — large gradients *suppress* saliency (upstream `--gradient_inv`; `small_value = 1e-8` guards zero division; a zero-gradient slot has denominator exactly `1e-8`, giving a huge metric that is **never pruned**).

**Step 4: mask and zero out:**

- unstructured (`prune_n = 0`): stable-sort `metric` **per row**, prune the `int(in · sparsity)` smallest columns of each row;
- N:M (`prune_n > 0`): require `0 < prune_n < prune_m`, `in % prune_m == 0`, `sparsity == 0.5`; split columns into blocks of `prune_m` and prune the `prune_n` smallest per block (same window rule as problem 581, but applied to the fused metric).

Return a dict (keys follow problem 581, plus `grad_l2`):

| Key | Meaning |
| --- | --- |
| `metric` | fused saliency matrix, `(out, in)` |
| `grad_l2` | `G` from step 2, `(out, in)` |
| `mask` | boolean mask, `True` = pruned |
| `pruned_weight` | weight with pruned entries zeroed |
| `sparsity` | realized sparsity = mask mean |

## Worked example

```
W = [[1, 2, 3, 4],
     [4, 3, 2, 1]]
A = [[1, 0, 1, 0],
     [1, 1, 0, 0]]      # 2 tokens → mean-based s_row = [1, 0.5, 0.5, 0]
grads (2 samples, scale = 1):
  sample1 = [[.1,0,0,0],[0,0,0,.2]]
  sample2 = [[.1,0,0,0],[0,0,.3,0]]
```

- `G`: row 0 `[0.1√2, 0, 0, 0]`, row 1 `[0, 0, 0.3, 0.2]`;
- `√s_row = [1, 1/√2, 1/√2, 0]`;
- `metric` row 0 = `[1+0.141421, 1.414214, 2.121320, 0]` → prune `{3, 0}`;
- `metric` row 1 = `[4, 2.121320, 2.014214, 0.2]` → prune `{3, 2}`;
- `pruned_weight = [[0, 2, 3, 0], [4, 3, 0, 0]]`, realized sparsity 0.5.

**`scale` is a real decision point**: changing it to 10 turns row 0 into `[2.428427, 1.414214, 2.121320, 0]` and the pruned set from `{3, 0}` to `{3, 1}` — after addition, different scaling factors shift the relative weight of the two terms.

**gradient_inv flip example**: `W = [[10, 8]]`, `A = [[1, 1]]`, single gradient `[1, 20]`: non-inv gives `[11, 28]` → prune column 0; inv gives `[10/1, 8/20] = [10, 0.4]` → prune column 1. A small-magnitude, large-gradient weight gets pruned first under inv.

## Boundaries

- `weight`/`activations` share one dtype (float32 or float64) with matching columns;
- `grad_samples` is 3-D with ≥ 1 sample and trailing shape equal to `weight.shape`;
- `sparsity ∈ [0, 1]`; `scale > 0`; `prune_n, prune_m ≥ 0`;
- N:M mode additionally requires `0 < prune_n < prune_m`, `in % prune_m == 0`, `sparsity == 0.5`;
- compute in float64 (upstream accumulates in fp16; judging compares at high precision);
- ties broken by stable sort (smaller index first); violations raise `ValueError`.

## Boundary with existing problems

- 581 `sparsity-wanda-activation-metric`: Wanda term only — no gradient fusion, no inv variant, no `scale`;
- 474 `sparsity-sparsegpt-reconstruct`: SparseGPT is OBS-style columnwise reconstruction, not score-then-rank;
- 218 `sparsity-nm-2-4`: tests the 2:4 window itself; here 2:4 is only the scope of the fused metric;
- 217 `sparsity-magnitude-prune`: pure magnitude with a global threshold — neither rowwise counting nor fusion applies.
