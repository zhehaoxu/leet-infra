# Multimodal: LLaVA-PruMerge CLS-Attention Selection and Cosine Residual Merge

## Task

Inside the CLIP vision tower, before the projector, use the CLS attention over patches to select a few "key patches", assign every remaining patch to the key patch it is most similar to by **cosine similarity**, and **merge their features residually** into that key patch, keeping only about `1/8` of the visual tokens.

This extracts the selection and merge steps of LLaVA-PruMerge (Shang et al., arXiv:2403.15388) from the official implementation. It is **not the complete method**: the patch features and the CLS-to-patch attention are supplied by the caller and no model forward runs here. The bilingual starter contract is the single reference for parameters, shapes, numeric limits, return order, and exceptions. The rules below define the computation.

## Selection rules

Let `N` be the patch count, `C` the feature dimension, and `k` the number of kept patches.

1. `k = int(N * reduction_ratio)`: the value **truncates** and does not round, so `int(5 * 0.5) = 2` and `int(3 * 0.5) = 1`. Raise `ValueError` when the truncation yields `0`.
2. Keep the `k` patches with the largest `cls_attn` values. Ties resolve as "value descending, then index ascending"; the kept indices must finally be emitted **in ascending order**, the merged-away indices likewise, and the two groups are disjoint and together cover `0..N-1`.

## Merge rules

Run the following per row, where `i` enumerates the ascending pruned patches and `j` the ascending kept patches (the column of `kept_indexs`):

1. The normalization denominator is `max(||x||_2, eps)` with `eps` defaulting to `1e-12`, matching `F.normalize`'s floor semantics. A zero vector therefore normalizes to all zeros, giving a cosine of `0` instead of `NaN`.
2. Similarity is `sim[i,j] = <u_i, u_j>`, where `u` is the unit vector from step 1.
3. Assignment is `cluster[i] = argmax_j sim[i,j]`, taking the **first** (smallest) `j` on ties.
4. `p[i,:]` is the stable softmax of `sim[i,:]` over the **whole kept set**, and then
   $$
   \mathrm{weight}[i] = p[i,\mathrm{cluster}[i]] \cdot C^{-1/2}.
   $$
   The scaling happens **after** the softmax and uses the feature dimension `C`, not the token count. Weights are **not renormalized** inside a cluster, so a cluster's weights usually do not sum to `1`.
5. The merge is a residual update:
   $$
   \mathrm{merged}[j] = x_{\mathrm{kept}[j]} + \sum_{i:\,\mathrm{cluster}[i]=j} \mathrm{weight}[i]\cdot x_{\mathrm{pruned}[i]}.
   $$
   A kept patch with no assigned pruned patch is **returned verbatim**. With `reduction_ratio = 1` the pruned set is empty and the output is just the ascending original patches.

`scores[i] = sim[i, cluster[i]]`, the maximum cosine of each pruned patch, is returned as a judgable ledger.

## Example 1: one cluster of three plus an empty cluster

All floating tensors use float64, `N = 5`, `C = 2`, `reduction_ratio = 0.5`, `eps = 1e-12`, hence `k = int(5 * 0.5) = 2`.

```text
image_features = [[[1, 1], [1, 0], [2, 0], [0, 1], [1, -1]]]
cls_attn = [[0.2, 0.9, 0.4, 0.7, 0.1]]

merged        = [[[2.956244, -0.215264], [0.000000, 1.000000]]]
kept_indexs   = [[1, 3]]
pruned_indexs = [[0, 2, 4]]
cluster       = [[0, 0, 0]]
weights       = [[0.353553, 0.516936, 0.568818]]
scores        = [[0.707107, 1.000000, 0.707107]]
```

The two largest attention values belong to patches `1` and `3`, which become kept columns in ascending order. All three pruned patches collapse onto column `0`: patch `0` ties at cosine `0.707107` against both columns and takes the first one, so its weight is the softmax value `0.5` at that column times `C ** -0.5 = 0.707107`. Column `1` receives nothing and is byte-identical to the original feature `[0, 1]` of patch `3`. The three weights sum to roughly `1.439307`, not `1`, which is the evidence that the softmax covers the whole kept set while the cluster is never renormalized.

## Example 2: truncation, a zero vector, and `C ** -0.5`

All floating tensors use float64, `N = 4`, `C = 4`, `reduction_ratio = 0.25`, `eps = 1e-12`, hence `k = int(4 * 0.25) = 1`.

```text
image_features = [[[1, 1, 0, 0], [0, 0, 0, 0], [2, 2, 0, 0], [1, 0, 1, 0]]]
cls_attn = [[0.6, 0.6, 0.9, 0.2]]

merged        = [[[3.000000, 2.500000, 0.500000, 0.000000]]]
kept_indexs   = [[2]]
pruned_indexs = [[0, 1, 3]]
cluster       = [[0, 0, 0]]
weights       = [[0.500000, 0.500000, 0.500000]]
scores        = [[1.000000, 0.000000, 0.500000]]
```

With a single kept column the softmax degenerates to `1`, so every weight is exactly `C ** -0.5 = 0.5`. Patch `1` is a zero vector: the denominator is floored by `eps`, its cosine is `0`, no `NaN` appears, and its contribution is `0.5` times the zero vector. The merge gives `[2, 2, 0, 0] + 0.5 * ([1, 1, 0, 0] + [0, 0, 0, 0] + [1, 0, 1, 0]) = [3, 2.5, 0.5, 0]`.

All displayed outputs were dumped from actual reference execution and checked against an independent oracle; values are rounded to six decimal places.
