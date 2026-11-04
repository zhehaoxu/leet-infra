# Notes: AdaMSS Adaptive Multi-Subspace Scatter Adaptation

## Problem

**AdaMSS** (Adaptive Multi-Subspace orthogonal fine-tuning, NeurIPS 2025, official repo `jzheng20/NeurIPS2025-AdaMSS`, merged into HF PEFT `tuners/adamss`) trains the SVD skeleton **in per-group segments**: SVD the bias-augmented weight \(W' = [W \mid b]\ (out, in{+}1)\); the right-factor projection \(\mathrm{newB} = V_r'^\top\ (r, in{+}1)\) is **frozen**, and the left factor \((out, r)\) is not trained wholesale — instead the \(r\) SVD components (out-dim directions) are clustered into \(K\) groups, each with an independent low-rank subspace

$$A_i\ (\mathrm{rank}_i, r),\quad B_i\ (\mathrm{seg}_i, \mathrm{rank}_i),\quad \mathrm{rank}_i = \max(1, \min(\mathrm{seg}_i,\ \mathrm{subspace\_rank}))$$

Subspaces have **heterogeneous ranks** and may be frozen entirely — the "adaptive multi-subspace" in the name.

The deterministic `adamss forward` chain (a ones column is appended to the input when bias is present; **the output dim stays out_features**):

```
x_aug     = [x | 1]                          # (..., in+1)
projected = x_aug @ newB.T                   # (..., r)   frozen projection
h_i       = projected @ A_i.T                # (..., rank_i)
y_i       = h_i @ B_i.T                      # (..., seg_i)
cat       = [y_0, y_1, ...]                  # (..., Σ seg_i)
delta[..., scatter_index] = cat              # scatter into out-dim rows
```

The merge trunk (shared by `get_delta_weight` / `get_delta_bias`):

```
combined = cat([B_i @ A_i], axis=0) @ newB    # (Σ seg_i, in+1)
dW_full[scatter_index, :] = combined          # (out, in+1) augmented delta
weight delta = dW_full[:, :-1]                # bias delta = dW_full[:, -1]
```

**The bias is the last COLUMN (the augmented ones column), not an extra row.** scatter_index is built at the end of the official `update_layer` by concatenating each segment's indices in **trained-subspace order**; rows of untrained groups stay zero in every delta. KMeans/SVD/PCA/QR are all done offline — the anchors pin only the deterministic matrix accounts.

## Rules

- `mode="build_scatter_index"`: given seg (group -> component indices, partitioning \(0..r{-}1\)) and train_idx, concatenate in trained-group order and return the 1-D row indices.
- `mode="rank_clamp"`: given seg_lens and subspace_rank, return per-segment \(\max(1, \min(L, \mathrm{subspace\_rank}))\) (the >=1 floor guards empty segments / zero rank).
- `mode="forward_delta"`: the full chain; with `with_bias=True` a ones column is appended to the input and folded through newB's bias column, with the output dim unchanged.
- `mode="get_delta_full"`: the augmented delta \((out, in{+}1)\) (bias column not stripped); rows outside scatter_index stay zero.
- `mode="split_delta"`: split into (weight, bias) — with_bias takes `[:, :-1]` and the last column; otherwise the full matrix plus a zero bias vector.
- newB/A_i/B_i/seg/train_idx are all explicit inputs; float64 throughout.

## Example

\(out=6, in=5, r=5\), components clustered into 3 groups \([0,1],[2,3],[4]\), subspace_rank=2, all trained:

```
rank_actual = [2, 2, 1]            # max(1, min(seg, 2))
scatter_index = [0, 1, 2, 3, 4]
forward_delta(x, newB, A, B, sc, 6) -> (2,3,6) = x @ dW.T   # cross-check
get_delta_full(newB_b, A, B, sc, 6) -> (6, 6)                # augmented form
split_delta(dW_full, True) -> (dW_full[:, :-1], dW_full[:, -1])
```

Partial training (group 1 skipped): scatter_index = [0, 1, 4]; rows 2 and 3 stay zero.

## Constraints

- Forward and merge are mathematically the same source: \(x\, @\, W_\Delta^\top\ (+\ b_\Delta) =\) forward (cross-checked in tests).
- rank_clamp: takes the segment length when subspace_rank exceeds it; 1 for an empty segment.
- The output dim always equals out_dim (bias is folded on the input side, not output + 1).
