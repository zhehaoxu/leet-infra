# KV Cache: Palu Grouped Rank-Budget Allocation

## Problem

Palu (2407.21118) compresses the KV projection weights themselves: the rows of `k_proj`/`v_proj` are split into head groups and **all heads in a group share one low-rank basis**, so a group's rank \(r_i\) is exactly the latent width the cache has to store (upstream concatenates the latents into `sum(ranks)` channels, see `svd_linear.py::HeadwiseLowRankModule`). The layer's keep ratio is \(\sum_i r_i / (G\cdot \texttt{lr\_group\_dims})\) — how the rank is split decides how many bytes are stored.

Group sensitivities differ wildly, and splitting the rank evenly wastes it on unimportant groups. The fisher branch of `rank_search.py` therefore **splits the total budget proportionally to Fisher information**:

\[
G=\frac{\text{rows}}{\texttt{head\_group\_size}},\qquad
f_i=\operatorname{mean}\bigl(\text{fisher}[\text{whole block of group } i]\bigr)
\]

\[
r^{\text{float}}_i=\frac{T\,f_i}{\sum_j f_j},\qquad
T=\texttt{ratio}\cdot \texttt{total\_rank},\qquad
\texttt{total\_rank}=G\cdot \texttt{lr\_group\_dims}
\]

A fractional rank cannot build a basis, so three more steps follow: a capped floor, a **smallest-remainder-first** round-robin of +1 grants, and the block alignment \(\max(1,\operatorname{round}(r/b))\cdot b\). This problem is exactly that **budget ledger** — the SVD, the whitening, the Hadamard rotation and the latent quantization are out of scope.

Three details that are easy to get wrong:

1. grants go out in **ascending** remainder order, not largest-remainder: upstream literally writes `sorted(..., key=rank_float - floor)`;
2. a group stops at \(\texttt{lr\_group\_dims}\): full groups are skipped while the cursor keeps moving, so the +1 grants can span many passes;
3. after alignment there is **no re-capping** and no check that `need` was consumed, so `rank_sum` can land either above or below \(T\).

## Task

Implement `solve(fisher, param_ratio_target, head_group_size, lr_group_dims, block_size=32)`:

- `fisher`: `(G*head_group_size, lr_group_dims)` floating tensor, the per-element sensitivity weights of the low-rank basis;
- `param_ratio_target`: \(\rho\in(0,1]\); `head_group_size`, `lr_group_dims`, `block_size`: positive integers (`bool` counts as illegal).

All arithmetic inside is float64; floating outputs fall back to the input dtype; `fisher` must not be written into. Return a 6-key dict:

- `rank_float`: `(G,)`, input dtype, \(T f_i / \sum_j f_j\);
- `rank_floor`: `(G,)` int64, \(\min(\texttt{lr\_group\_dims},\ \lfloor r^{\text{float}}_i\rfloor)\);
- `rank_topup`: `(G,)` int64, `rank_floor` plus `need` grants of 1 handed out in ascending remainder order, `need = floor(T - Σ rank_floor)`;
- `rank_aligned`: `(G,)` int64, \(\max\bigl(1,\ \operatorname{round}_{\text{half-even}}(\texttt{rank\_topup}_i / \texttt{block\_size})\bigr)\cdot \texttt{block\_size}\);
- `rank_sum`: 0-dim int64, \(\sum_i \texttt{rank\_aligned}_i\);
- `stored_ratio`: 0-dim, input dtype, `rank_sum / total_rank`.

Raise `ValueError` on: `fisher` not a 2-D tensor, not floating point, rows not divisible by `head_group_size`, columns different from `lr_group_dims`, non-positive sum of the group Fisher values; `param_ratio_target` outside \((0,1]\); any of the three integer parameters not a positive integer.

## Example

```
fisher          # (16, 32): 4 groups x 4 rows each, group values 4.0 / 3.0 / 2.0 / 1.0
head_group_size = 4        # G = 16 // 4 = 4
lr_group_dims   = 32       # total_rank = 4 * 32 = 128
param_ratio_target = 0.5   # T = 128 * 0.5 = 64.0
block_size      = 8

f            = [4.0, 3.0, 2.0, 1.0]          # mean over each (4, 32) block, sum_f = 10.0
rank_float   = [25.6, 19.2, 12.8, 6.4]       # 64.0 * f / 10.0
rank_floor   = [25, 19, 12, 6]               # sum = 62, leftover T - sum = 2.0
remainder    = [0.6, 0.2, 0.8, 0.4]          # ascending -> grant order [1, 3, 0, 2]
rank_topup   = [25, 20, 12, 7]               # need = floor(2.0) = 2: groups 1 and 3 get +1
rank_aligned = [24, 16, 16, 8]               # round(3.125)=3, round(2.5)=2, round(1.5)=2, round(0.875)=1
rank_sum     = 64                            # 24 + 16 + 16 + 8
stored_ratio = 0.5                           # 64 / 128
```

The two single-rank grants go to groups 1 and 3, the **smallest** remainders, not to groups 2 and 0 with the largest ones — the opposite of textbook largest-remainder apportionment, and literally what upstream does. `stored_ratio` landing back on 0.5 is a coincidence too: alignment lifts 12 to 16 and pushes 25 down to 24, flattening the carefully computed proportions; in the tests `ratio = 0.5` aligns to `rank_sum = 20` (above \(T=12\)) while `ratio = 0.55` aligns to 64 (below \(T=70.4\)) — it drifts in both directions.

## Constraints

- **Group Fisher is a whole-block mean**: the mean is taken over each group's entire `(head_group_size, lr_group_dims)` block, not over rows, and not column-then-row. In the example every row is constant so the variants coincide by accident; they do not in general.
- **Grants go in ascending remainder order**: the sort key is `rank_float - rank_floor` with the **capped** `rank_floor`, ascending, ties to the smaller group index (stable sort).
- **Two caps**: `rank_floor` is clipped by `min(..., lr_group_dims)`; during the round-robin a group already at `lr_group_dims` is skipped while the cursor keeps moving, so `need` can take several passes to hand out.
- **`need = floor(T - Σ rank_floor)`**: a leftover below 1 is never granted, and `check_fractional_budget` pins the topped-up total at 70 rather than 71.
- **Alignment: half-to-even, one block guaranteed, never re-capped**: `20/8 = 2.5` and `12/8 = 1.5` both go to the even 2 -> 16, and a group with `rank_topup = 0` is still raised to one `block_size`; alignment does not clip back, so `rank_aligned` may exceed `lr_group_dims` (when it is not a multiple of `block_size`), and `stored_ratio` therefore is not `param_ratio_target`.
- **Dtypes and shapes**: `rank_float`/`stored_ratio` use the input dtype, the other four keys are int64, `rank_sum` and `stored_ratio` are 0-dim tensors, and `fisher` must come back unchanged.
