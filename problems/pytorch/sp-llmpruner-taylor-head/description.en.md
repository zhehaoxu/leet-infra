## Problem

LLM-Pruner (2305.11627) prunes at **dependency-group** granularity, not single weights: in multi-head attention, a head of q/k/v is coupled with the matching row block of o_proj — touching the head means touching all four. It scores groups with first-order Taylor saliency `salience = w · g` (elementwise), approximating the direct loss impact of removing each parameter.

Given the four attention projection matrices and their gradients, reproduce the per-head scalar importance and head-removal selection. Inputs:

- `wq` / `wk` / `wv` / `wo`: four `(d, d)` weight matrices (MHA layout: head `h` owns columns `[h·head_dim, (h+1)·head_dim)` of q/k/v and the same rows of o);
- `gwq` / `gwk` / `gwv` / `gwo`: matching gradients (same shapes, same dtype);
- `head_dim`: per-head dimension; `d` must be divisible by it;
- `mode`: `"param_first"` or `"vectorize"` (upstream `--taylor`);
- `group_reduction`: `"sum"` or `"mean"` (upstream `--grouping_strategy`);
- `num_prune`: how many heads to remove.

## Rules

**Step 1: elementwise Taylor saliency.** `salience_X = w_X · g_X` for each of the four matrices.

**Step 2: row/column reduction** (upstream attention branch: o_proj is an out-channels dependency, q/k/v are in-channels):

| mode | o_proj (rows) | q/k/v (cols) |
| --- | --- | --- |
| `param_first` | `\|salience_o\|.sum(dim=1)` | `\|salience_X\|.sum(dim=0)` |
| `vectorize` | `\|salience_o.sum(dim=1)\|` | `\|salience_X.sum(dim=0)\|` |

`param_first` takes absolute values first (magnitude evidence); `vectorize` sums first — **signs cancel** within a block and the block score may collapse to 0.

**Step 3: head-block aggregation.** Split each `(d,)` vector into `head_dim`-blocks and sum per block, giving per-head block contributions `(n_heads,)` for o (rows) and q/k/v (columns) separately.

**Step 4: cross-source reduction and selection.** Stack the four contributions into `(4, n_heads)` and reduce with `group_reduction` (`sum` / `mean`; they differ by a constant factor of 4 — same selection, different values), yielding `head_importance`. Sort ascending with **stable** ordering and remove the `num_prune` smallest heads; ties go to the smaller index.

Return dict:

| Key | Meaning |
| --- | --- |
| `head_importance` | per-head scalar importance, `(n_heads,)` float64 |
| `per_source` | per-source (o/q/k/v) head-block contributions, `(4, n_heads)` |
| `pruned_heads` | ascending list of removed head indices |
| `n_heads` | `d / head_dim` |

## Worked example

```
w = [[ 1,  2,  3,  4],
     [ 5,  6,  7,  8],
     [ 9, 10, 11, 12],
     [13, 14, 15, 16]]   # all four matrices; grads all 1; head_dim=2
```

- `salience = w`; o row sums `[10, 26, 42, 58]` → head blocks `[36, 100]`;
- q/k/v column sums `[28, 32, 36, 40]` → head blocks `[60, 76]`;
- `head_importance = [36+60+60+60, 100+76+76+76] = [216, 328]`;
- `num_prune=1` → remove head 0.

**vectorize cancellation example**: only q nonzero; head-0 columns `[1,-3,1,-3]`, head-1 columns `[2,-2,2,-2]`:
- `param_first`: column sums of `|·|` are `[6, 8]` → `[6, 8]`;
- `vectorize`: column sums are `[0, 0]` → `[0, 0]` — same input, completely different importances.

**Selection-flip example** (`head_dim=1`): head 0's q column `[3, -3]` (abs-sum 6, abs of sum 0), head 1's q column `[2, 0]`: `param_first` gives `[6, 2]` → remove head 1; `vectorize` gives `[0, 2]` → remove head 0. The timing of `abs` decides which head dies.

## Boundaries

- The four weights and four gradients pairwise share shape `(d, d)` and one dtype (float32/float64); weights and gradients share dtype;
- `d % head_dim == 0` in both row and column directions, `head_dim ≥ 1`;
- MHA layout requires `d_out == d_in` (o rows align with q/k/v columns);
- `num_prune ∈ [0, n_heads]`; `mode` / `group_reduction` enum-checked;
- stable sort ties to smaller index; `num_prune=0` returns an empty list; violations raise `ValueError`.

## Boundary with existing problems

- 470 `sparsity-head-importance`: head importance as `||W_h||_F · ||X_h||_F` (activation norms, no gradients); here Taylor products with group coupling;
- 582 `sparsity-owl-outlier-layerwise`: OWL allocates sparsity across layers; this problem scores coupled groups;
- 472 `sparsity-ste-mask-backward`: STE is the straight-through gradient path for masks, no group selection;
- 473 `sparsity-movere-reactivation`: MoveOver reactivates pruned channels — the opposite direction.
