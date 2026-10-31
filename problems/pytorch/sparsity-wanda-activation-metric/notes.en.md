# Solution Notes · Wanda Activation-Weighted Magnitude Pruning

## In one line

Importance is \(=|W_{ij}|\cdot\sqrt{\mathbb{E}_t[X_{t,j}^2]}\); each output row prunes its `int(in·s)` smallest entries, and N:M competes inside each block.

## Why multiply by activation RMS

A layer output is \(y_i=\sum_j W_{ij}x_j\). Removing one term perturbs it by roughly \(|W_{ij}|\cdot|x_j|\), so importance is naturally a product of the weight and the **typical magnitude of that input channel**. The RMS over calibration tokens expresses that magnitude, which is exactly upstream's `W_metric`. This is why a column with weight 3 but an all-zero activation is pruned first: it contributes to no input. Upstream accumulates squared norms in `scaler_row` and takes the square root inside `W_metric`; this problem states the equivalent "mean square, then root" form.

## Three conventions

1. **k truncates**. `int(7*0.5)=3`. Upstream uses `int(shape[1]*sparsity)`; do not substitute `round`. The existing `sparsity-rowwise-prune` uses `round`, so the two problems contrast deliberately.
2. **Ties go to the smaller index**. Upstream's stable ascending sort keeps column order, so the first k entries prune the smaller indices. This problem pins that as an explicit rule and tests it with all-equal inputs.
3. **N:M accepts only 0.5**. With `prune_n>0`, columns are grouped into blocks of `prune_m` and each block prunes its `prune_n` smallest; `main.py` asserts `sparsity_ratio == 0.5` for `2:4`/`4:8`. This problem additionally requires `in_features % prune_m == 0`, because upstream calls `topk(prune_n)` on a ragged tail block where the operation degenerates.

## Grading points

- `metric` broadcasts **per column**, not per row, and matches the shape of `weight`;
- `mask` means True = pruned, the same as upstream `W_mask`;
- `pruned_weight` zeroes only pruned positions and keeps signs of retained entries;
- the achieved sparsity is measured from the mask: exactly 0.5 in structured mode, `int(in·s)/in` otherwise;
- inputs are never mutated.

## Boundaries with neighbouring problems

- `sparsity-magnitude-prune`: global threshold, \(|W|\) only.
- `sparsity-rowwise-prune`: per row but \(|W|\) only, and uses `round`.
- `sparsity-nm-2-4`: only the 2:4 block-wise largest selection.
- `sparsity-sparsegpt-reconstruct`: uses the inverse Hessian to compensate errors; Wanda performs no compensation at all.

## Source

`locuslab/wanda` (arXiv 2306.11695):

- `lib/prune.py::prune_wanda`: `W_metric = torch.abs(weight) * torch.sqrt(scaler_row.reshape((1,-1)))`; `sort(..., stable=True)`; `int(W_metric.shape[1] * args.sparsity_ratio)`; N:M branch `topk(tmp, prune_n, dim=1, largest=False)`;
- `lib/layerwrapper.py::WrappedGPT.add_batch`: `scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / nsamples`;
- `main.py`: `assert sparsity_ratio == 0.5` for structured pruning.

**Added by this problem**: explicit inputs instead of running a model; ties-to-smaller-index as a hard rule; the `in_features % prune_m` check; achieved sparsity measured from the mask.
