# Solution Notes · OWL Outlier-Ratio Layer-Wise Sparsity Allocation

## In one line

Concatenate `|W|·RMS` across all Linears of a layer, count how many entries exceed "mean × hyper_m" to get the outlier ratio; min-max normalise ratios onto `[0, 2λ]`, subtract the mean and align to `1-s` to get per-layer density; then prune `int(in·s_l)` smallest per row inside each layer.

## Why allocate budget by outlier ratio

With a fixed global sparsity the split across layers is a free variable. OWL's observation: the heavier the activation tail of a layer, the more its metric distribution is pulled up by a few large values, and the larger the fraction above `mean·hyper_m`; such layers are more sensitive to error and should be pruned less. Outlier ratio therefore serves as the layer score; after min-max normalising it, density is distributed by score. Subtracting the mean guarantees that **the average density returns to `1-s`**, i.e. the total budget is unchanged.

## Three easy mistakes

1. **Concatenate per layer, not per Linear**. `layer_wmetric.append(W_metric)` then `torch.cat([flatten(x) ...])`, so mean and threshold come from the whole layer; each Linear merely shares the resulting layer sparsity.
2. **Strictly greater**. `(W > max_shred).sum()`; equality does not count. For `[1,1,1,9]` the mean is 3 and the threshold 9, so 9 is not an outlier.
3. **What the 0.01 floor means**. `if layer_sparsity_ratio<=0: layer_sparsity_ratio=0.01` sets sparsity to 1%, not density to 1%. There is no upper clamp: when `int(in·s)` exceeds `in` the slice returns all columns, equivalent to pruning the whole layer.

## Grading points

- `outlier_ratio`, `density`, `layer_sparsity` are float64 with shape `(L,)`;
- `metric`/`pruned_weight` use the input dtype and nesting, `mask` is bool with True = pruned;
- all Linears of a layer share the layer ratio, each computing its own `int(width·s_l)`;
- `layer_sparsity.mean() == sparsity` when the 0.01 floor is not triggered;
- identical ratios across layers raise `ValueError` instead of returning NaN.

## Boundaries with neighbouring problems

- `sparsity-wanda-activation-metric`: single-layer metric plus row-wise mask; this problem adds **cross-layer budget allocation** on top.
- `sparsity-pyramid-layer-budget` (kvcache): linear budget by depth for KV cache; OWL allocates **weight-pruning** budget by outlier statistics.
- `sparsity-rowwise-prune`: row-wise pruning with one sparsity for all layers.

## Source

`luuyin/OWL` (arXiv 2310.05175):

- `lib/prune_all.py::prune_wanda_outlier`: `W_metric = |W| * sqrt(scaler_row.reshape((1,-1)))`; `layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])`; `out_ratio_layer = check_outlier_mean(layer_wmetric, args.Hyper_m)`;
- `check_outlier_mean`: `max_shred = torch.mean(W)*threshold`, `count += (W > max_shred).sum()`, `outlier_ratio = count/total_params*100`;
- budget: `(r - r.min()) * (1/(r.max()-r.min()) * args.Lamda*2)`, `- np.mean(...) + (1-args.sparsity_ratio)`, `layer_sparsity_ratio = 1 - all_layer_ratio[i]`, `<=0 → 0.01`;
- mask: `sort(W_metric, dim=-1, stable=True)` then the first `int(cols*layer_sparsity_ratio)`.

Defaults from `main.py`: `Lamda=0.08`, `Hyper_m=3`; `sparsity_ratio` must be given explicitly.

**Added by this problem**: explicit `[layer][Linear]` structure; ties-to-smaller-index as a hard rule; identical ratios raise an error instead of producing NaN; per-layer `density` is returned for comparison.
