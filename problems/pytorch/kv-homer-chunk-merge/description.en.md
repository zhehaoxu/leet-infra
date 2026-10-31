# KV Cache: HOMER Hierarchical Chunk Merging

## Problem

Prefill memory grows linearly with the context. HOMER slices the context into \(2^h\) chunks as tree leaves: each leaf runs a few layers and is pruned to a target length by attention, then sibling nodes are merged in pairs and pruned again, until the root holds a single chunk. Each level only touches "half as many, half as long" chunks, so peak memory depends on the chunk length, not on the document length.

A single node does two things: **merge** and **prune**.

Merge (`Chunk.merge`): two chunks share one affix — the first `prefix_len` tokens and the last `suffix_len` tokens sit at the same place in every chunk. When merging, the two children's prefixes are **averaged elementwise**, the suffixes likewise, and the real context in between is simply concatenated front to back. The order is `[prefix mean, left context, right context, suffix mean]`, and the length is \(M = M_L + M_R - \text{prefix} - \text{suffix}\).

Prune (`_merge_power_max_last_calibrated`): take the last row of `metric`, max over heads, and subtract a position-dependent calibration bias

\[
\text{sig}(j) = \max_h \text{metric}[h, -1, j] - \text{bias}[\text{rel}(j)],\qquad \text{rel}(j) = (M-1) - j
\]

`rel` is "how far this token is from the end", so `bias[0]` lands on the last position. After the subtraction, positions whose `reduction_mask` is False (the affix, protected) are set to `+inf`, and `topk(largest=False)` removes the `num_tokens_to_reduce` smallest — the protected ones are `+inf`, so they can never be among the smallest.

## Task

Implement `solve(k_left, v_left, k_right, v_right, metric, bias, reduction_mask, num_tokens_to_reduce, prefix_len, suffix_len)`:

- `k_left`/`k_right`: `(M_L, H, D)` / `(M_R, H, D)`, NHD layout; each `v_*` has the same shape and dtype;
- `metric`: `(H, Q, M)`; **only the last row** `metric[:, -1, :]` is read;
- `bias`: `(R,)` with `R >= M`, indexed by distance from the end;
- `reduction_mask`: `(M,)` bool, True means prunable;
- `num_tokens_to_reduce`, `prefix_len`, `suffix_len`: non-negative integers with `prefix_len + suffix_len >= 1`.

Everything inside is float64 and the outputs use the input dtype.

Return a 4-key dict:

- `weights`: `(M,)`, the significance after bias subtraction, with `+inf` on the protected positions (upstream's return value carries exactly those `inf`s);
- `removed`: `(r,)` int64 ascending, the dropped indices (ties go to the smaller index);
- `merged_k` / `merged_v`: `(M-r, H, D)`, the merged cache in its original order.

With `num_tokens_to_reduce = 0` nothing is removed, the length stays `M`, and the weights are still computed.

Raise `ValueError` on: not 3-D, differing left/right dtype or a dtype other than float32/float64, head count or head_dim mismatch, k/v length mismatch, `metric` not `(H, Q, M)`, `bias` shorter than `M`, `reduction_mask` not a `(M,)` bool, arguments that are not non-negative integers, an all-zero or oversized affix, `r >= M`, or `r` larger than the number of prunable positions.

## Example

```
k_left  = [[10], [12], [1], [2], [3], [40]]        # (6,1,1): prefix 2 + context 3 + suffix 1
k_right = [[20], [30], [7], [8], [50]]             # (5,1,1): prefix 2 + context 2 + suffix 1
prefix_len = 2, suffix_len = 1                     # merged M = 6 + 5 - 3 = 8
v_left, v_right = k_left, k_right                  # K and V coincide in this example
metric = [[[0.9, 0.8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]]     # (H,Q,M) = (1,1,8)
bias   = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
reduction_mask = [F, F, T, T, T, T, T, F]
num_tokens_to_reduce = 3

merged_k = [(10+20)/2, (12+30)/2, 1, 2, 3, 7, 8, (40+50)/2]
         = [15, 21, 1, 2, 3, 7, 8, 45]
rel      = [7, 6, 5, 4, 3, 2, 1, 0]
weights  = metric[0,-1,:] - bias[rel]
         = [0.55, 0.50, -0.15, 0.00, 0.15, 0.30, 0.45, 0.60]
         = [inf, inf, -0.15, 0.00, 0.15, 0.30, 0.45, inf]   # protected -> +inf
removed  = [2, 3, 4]                                        # the three smallest
merged_k = [15, 21, 7, 8, 45]                               # what is left
```

The three lowest weights, −0.15, 0.00 and 0.15, sit at positions 2, 3 and 4 and are exactly the ones removed; positions 0, 1 and 7 start at 0.55, 0.50 and 0.60 (not among the three smallest anyway) and are additionally shielded by `+inf`, so no `r` can ever drop them.

## Constraints

- **The affix is averaged, not duplicated**: both children hold the same prefix tokens, so concatenating would replicate the affix \(2^h\) times; averaging keeps exactly one copy.
- **The concatenation order** is `[prefix mean, left context, right context, suffix mean]`, never "left chunk then right chunk".
- **The bias counts from the end**: `bias[0]` applies to the last position and `bias[M-1]` to the first.
- **Protection is `+inf`, not 0**: pruning takes the smallest values, and 0 can still be the smallest; only `+inf` blocks it.
- **Ties go to the smaller index**: upstream's `torch.topk` leaves tie order undefined, so this problem fixes it.
- **`r = 0` prunes nothing**: the full merged `M` tokens come back and the weights are unchanged.
