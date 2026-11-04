# LLM-Barber Mask Rebuild

## Problem

After one-shot pruning the sparse model loses accuracy. LLM-Barber's fix: instead of going back to dense and re-pruning, perform an **exchange on the existing sparse pattern** — revive pruned-but-important weights (growth) and prune alive-but-unimportant ones (prune), rebuilding the mask while keeping sparsity unchanged.

You are given:

- `weights`: dense reference weights (dict[str, 2-D tensor], several matrices such as attention q/o and MLP fc1/fc2);
- `ref_weights`: the pruned sparse weights (**zero positions are exactly the current mask's pruned entries**);
- `scalers`: per-weight grad² Fisher running averages (collected upstream via block forward MSE backprop; given explicitly here, outside the judging scope).

## The three steps to implement

### 1. Metric (three choices)

Compute importance on a `combined` weight:

- with `retain_reconstruction=True`: `combined = where(mask, weights, ref_weights)` (pruned positions refill the dense value, alive positions keep the reconstructed sparse value);
- with `retain_reconstruction=False`: `combined = weights.clone()` (take the dense weight wholesale).

| metric | formula | meaning |
|--------|---------|---------|
| `WD` | \|combined\| · √scaler | magnitude × importance |
| `W` | \|combined\| | magnitude only |
| `D` | √scaler | Fisher importance only |

### 2. find_threshold (dual thresholds)

Call the pruned positions' metrics the growth set and the alive ones the prune set:

- sort growth descending and prune ascending; pair the first `min_len = min(sizes)` entries **position-wise**;
- `recons_num = Σ(growth[i] − prune[i] > 0)` (strictly greater; an exact 0 difference does not count);
- `idx = int(recons_num × threshold)`;
- `growth_threshold = growth_sort[idx]`, or `growth_sort[-1]` (last element) when `idx == growth count` (out of range);
- `prune_threshold = prune_sort[idx]`, or `prune_sort[-1]` when `idx == prune count`;
- if either side is empty return `(+inf, −inf)` — neither comparison fires and the mask stays unchanged.

Intuition: recons_num counts the pairs whose exchange is beneficial; threshold controls what fraction to exchange.

### 3. Exchange and rebuild

`granularity` determines the threshold scope:

- `block`: concatenate the growth/prune sets of **all** matrices and compute one global pair of thresholds;
- `layer`: one pair per matrix;
- `output1`: one pair per output row;
- `input1`: one pair per input column.

Then for each matrix (block reuses the global thresholds):

- `growth_indices = metric > growth_threshold` (strictly greater);
- `prune_indices = metric < prune_threshold` (strictly less);
- `mask[growth_indices] = False` (revive), then `mask[prune_indices] = True` (prune);
- `rebuilt = where(mask, 0, combined)`; return the overall sparsity `Σmask / numel`.

## Input constraints

- `weights` / `ref_weights` / `scalers` are non-empty dicts with identical non-empty string keys and shape-consistent 2-D float tensors;
- `scalers` are non-negative (Fisher quantities are grad² running averages);
- `metric ∈ {WD, W, D}`; `granularity ∈ {block, layer, output1, input1}`;
- `threshold` is a finite number in [0, 1] (bools do not count).

Return `dict(masks={name: bool tensor}, rebuilt={name: weight}, sparsity=float, thresholds={name: (growth, prune)})`.

## Hints

- The exchange order matters: revive first, prune second — a position hit by both sets ends up pruned.
- The official N:M structured branch (per-row find_threshold plus interleaved in-block topk) is out of scope.
- Fisher collection (grad² running average from block forward MSE backprop) is outside the judging scope; scalers are direct inputs.
