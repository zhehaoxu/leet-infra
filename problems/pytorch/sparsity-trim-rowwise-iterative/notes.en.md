# Solution Notes · TRIM Row-Wise Sparsity via Output-Cosine Feedback

## In one line

Use the cosine between pruned and unpruned row outputs as feedback, moving sparsity budget each round from rows whose output barely changes to rows whose output moves, then pick the best round over a learning-rate grid.

## Motivation

A uniform row sparsity implicitly assumes every row tolerates the same fraction of deletions. TRIM starts from the observation that output rows react differently to the same fraction, so each row's sparsity should follow the **fidelity of its pruned output**. It introduces no new importance measure; it closes the loop in output space: when a row's cosine drops, reduce its sparsity (prune less) and hand the budget to rows whose cosine is still near 1.

## Algorithm

1. **Fixed ordering**. `sorted_idx` comes from a stable ascending sort of `|W|·sqrt(act_ms)`, and pruning is always a prefix, so each round only changes how deep to prune.
2. **Normalised delta**. `d=(q-min)/(max-min+1e-6)` maps per-row cosines into `[0,1]`; multiply by `2η` and subtract the mean before adding to \(x\). Subtracting the mean keeps the average budget unchanged; the following `x-mean(x)+target` pins it back to the target.
3. **Hard clamps**. \(x\) is clamped to `[0, 0.95]` before re-alignment; `0.95` is upstream's stability cap against pruning a whole row.
4. **Selection metric**. The flat layer cosine (`cosim_flat`) measures how well the pruned layer output keeps its direction; each round re-prunes and re-multiplies by `inputs`, recording `best_x` and `best_iter` only on strict improvement.
5. **Learning-rate scan**. The positive grid is tried in order and stops at the first non-improvement (the grid increases, so larger steps cannot help after that); the negative grid runs only if the positive grid never beat the baseline. `learning_rate` may be `None`.
6. **Post-processing**. The best \(x\) is clamped and re-aligned once more before building the mask; the returned `sparsities` are post-processed while `best_metric`/`learning_rate` refer to the pre-post-processing selection.

## Grading points

- each row of `keep_mask` prunes exactly `int(in·s_i)` entries (negative counts as 0, above 1 prunes everything);
- `sparsities.mean() == target_sparsity` (the second re-alignment restores it after clamps);
- `best_metric >= baseline_metric`, strictly greater whenever `learning_rate` is not None;
- `starting_var`/`ending_var` use the unbiased variance; `best_iter` is 0-based;
- inputs are not mutated; `pruned_weight` only changes pruned positions.

## Boundaries with neighbouring problems

- `sparsity-wanda-activation-metric`: one-shot Wanda with a uniform row sparsity; this problem iterates on top of that ordering to produce **different per-row sparsities**.
- `sparsity-owl-outlier-layerwise`: allocates budget **across layers**, uniform within a layer; this problem allocates **across rows**.
- `sparsity-movere-reactivation`: dynamic sparse training that changes masks via gradients; this is a one-shot post-training solve with no gradients.

## Source

`flobk/TRIM` (arXiv 2505.16743):

- `lib/trim.py::calculate_row_wise_sparsities`: positive/negative learning-rate grids, `best_metric`/`best_learning_rate`/`best_iter` updates and early stopping;
- `lib/trim.py::get_sparsities_iterative`: `unpruned_outputs = W @ inputs`, and `update_x` with `(qm-min)/(max-min+1e-6)`, `*learning_rate*2`, `-mean`, `x>0.95→0.95`, `x<0→0`, `x-mean+target`;
- `lib/trim.py::prune_matrix`: `num_prune = (width*sparsities).long()`, `mask[row, sorted_idx] = ~prune_mask`;
- `lib/trim.py::cosine_similarity*`: the `1e-8` norm guard and `clamp(-1,1)`;
- the TRIM branch of `lib/prune.py`: `W_metric = |W|*sqrt(scaler_row)`, `sorted_idx = sort(..., stable=True)[1]`, and the post-call `>0.95→0.95`, `<0→0`, `-mean+ratio`.

**Added by this problem**: `inputs` and `activation_ms` are explicit (upstream takes the last token for one and all-token statistics for the other); ties-to-smaller-index is a hard rule; only `cosim_flat` is supported; `out>=2` and `target∈(0,1)` are validated.
