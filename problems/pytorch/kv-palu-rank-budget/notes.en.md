# Solution Notes · Palu Grouped Rank-Budget Allocation

## In one sentence

Cut the `total_rank * ratio` budget into integer ranks in proportion to each group's Fisher weight: capped floor -> round-robin grants in **ascending** remainder order -> `block_size` alignment. Pure budget ledger, no decomposition at all.

## Why the ledger looks like this

The group sensitivity \(f_i\) is the mean of the group's whole block, the budget is \(T=\rho\cdot G\cdot L\) with \(L=\texttt{lr\_group\_dims}\), so

\[
r^{\text{float}}_i = T\,\frac{f_i}{\sum_j f_j},\qquad
r^{\lfloor}_i=\min\bigl(L,\ \lfloor r^{\text{float}}_i\rfloor\bigr)
\]

The proportional form guarantees \(\sum_i r^{\text{float}}_i = T\): rank follows sensitivity instead of being split evenly over heads. After the floor the shortfall \(d=T-\sum_i r^{\lfloor}_i\) is usually fractional, and handing it back one unit at a time gives `rank_topup`; the final \(\max(1,\operatorname{round}(r/b))\cdot b\) snaps the latent width to a multiple of \(b\) (upstream hardcodes 32 so the `sum(ranks)`-wide latent lines up with the quantization groups). Only the third of the three stages breaks the total.

## Four conventions that are easy to get wrong

1. **Ascending grants**: the sort key `rank_float - rank_floor` is ascending, so the groups with the **smallest** remainders get the grants. That is the opposite of the largest-remainder method, but upstream literally writes `sorted(..., key=rank_float - floor)` and the test copies it.
2. **Remainder after the clamp**: for a full group `rank_float` far exceeds \(L\), so after clamping the remainder is \(r^{\text{float}}_i - L\), a large positive number that sinks to the back of the ascending order. Computing the remainder before clamping gives a completely different grant order.
3. **`floor(d)` grants, full groups are skipped not refunded**: a leftover below 1 is simply dropped; each cursor step that lands on a full group just continues, so `need` can stretch over many passes.
4. **Alignment neither caps nor conserves**: `max(1, .)` guarantees one block even for a zero-rank group, `round` is half-to-even, and nothing clips back to \(L\); `rank_sum` can end up above or below \(T\), and `stored_ratio` is not `param_ratio_target`.

## What is judged

- All six keys, with exact dtypes: `rank_float`/`stored_ratio` follow the input dtype (tolerance 1e-5 for float32, 2e-9 for float64), `rank_floor`/`rank_topup`/`rank_aligned` are int64 vectors, `rank_sum` is a 0-dim int64;
- `rank_topup` must match "ascending order, skip full groups, exactly `floor(d)` grants" element by element - one misplaced unit fails;
- `rank_aligned` tests half-to-even (2.5 and 1.5 both go to 2) and the guaranteed single block for a zero-rank group at once;
- `rank_sum == Σ rank_aligned` and `stored_ratio == rank_sum / total_rank`, neither of which equals `ratio` in general;
- float64 inside, and `fisher` must come back untouched.

## Boundaries

- `ratio = 1.0`: \(T=\texttt{total\_rank}\); groups whose floor hits \(L\) are skipped and their grants move on, so every group can end up full (`check_cap_skips_and_cycles` fills all three).
- A single group with \(f_i=0\) is legal as long as the sum is positive: `rank_float`, `rank_floor` and `rank_topup` are 0, yet alignment still gives it one block.
- All-zero / non-positive total raises `ValueError`, as do row counts not divisible by `head_group_size` and column counts different from `lr_group_dims`.
- The `need > headroom` guard (`rank budget exceeds available headroom`) is unreachable for legal input: \(\texttt{need}\le T-\sum r^{\lfloor}\le G L-\sum r^{\lfloor}=\texttt{headroom}\), and headroom is an integer so the floor cannot cross it; it only stands guard against surprises such as NaN.

## Complexity

Block means cost \(O(G\cdot\texttt{head\_group\_size}\cdot L)\) (one pass over the input), sorting \(O(G\log G)\), and the round-robin at most \(O(G\cdot\texttt{need})\) cursor steps - within any \(G\) consecutive steps at least one grant lands, so full-group visits only occupy slots inside those \(G\); alignment and summation are \(O(G)\).

## Source

The rank search of `shadowpa0327/Palu` (arXiv 2407.21118), `search_method == "fisher"` branch:

- `palu/rank_search.py::rank_search` L105-168: L120 `module_rank = info.num_lr_groups * info.lr_group_dims`, L123 initial value `[info.lr_group_dims] * num_lr_groups` (that is the cap); L125 `fisher.reshape(num_lr_groups, -1, in_features)`, L129 `torch.mean(fisher[i])` (group Fisher = whole-block mean), L131 `fisher_sum`; L134 `target_rank = total_rank * param_ratio_target`, L142 `rank_float = target_rank * fisher[i] / fisher_sum`, L146 `min(select_result[name][i], math.floor(rank_float))`; L150 `indexes = sorted(indexes, key=rank_float - floor)`, L151 `dif`, L154-163 the `while dif > 0` round-robin (full groups `continue`); L164 the rounding, L165-166 `rank_sum / total_rank`.
- `palu/rank_search.py::rounding_search_result` L11-17: L11 `block_size=32`, L15 `max(1, round(ranks[i] / block_size)) * block_size` - Python's builtin `round` already is half-to-even.
- Group semantics in `palu/model/svd_llama/modeling_palu_llama.py::get_kv_info` L38-59: L40 `num_lr_groups = num_key_value_heads // head_group_size`, L42 `lr_group_dims = head_dim * head_group_size`, L44-54 non-divisible cases raise `ValueError`.
- The consumer of the ranks (out of scope here): `palu/decomposition.py::compress_model_svd` L228-259 -> `palu/model/modules/svd_linear.py::HeadwiseLowRankModule.from_linear` L206-235 (L212 reshapes the weight per group, L218 truncates each group's SVD to `ranks[i]`), L72 `nn.Linear(in_features, sum(ranks))` i.e. latent width = sum of ranks, and `fused_hadamard_matrix` L156-168, the Hadamard rotation paired with latent quantization.

**Additions of this problem** (not written upstream, or written differently, and pinned explicitly here): one module and one budget per call - upstream accumulates the Fisher of every k_proj/v_proj across all layers into one global `fisher_sum` at L131 - and the whole thing is folded into a single `(G*head_group_size, lr_group_dims)` matrix whose column width means `lr_group_dims` rather than upstream's `in_features`; `head_group_size`/`lr_group_dims`/`block_size` are explicit arguments; the grant count is pinned to `floor(dif)`, because upstream's inner loop only breaks on `dif == 0` and with a fractional `dif` it walks the whole list and over-grants; a `need > headroom` guard is added (upstream would spin forever once every group is full); equal remainders keep the smaller group index via a stable sort; the full shape/dtype/ratio/non-positive validation; and six intermediate outputs instead of one `logger.info` compression-ratio line.

## Related problems

- `kv-xkv-layer-svd-share`: also low-rank sharing, but shared across **layers' singular vectors**; here sharing is only among the heads of one group and no basis is computed.
- `kv-layer-budget-knapsack`: a knapsack optimum over per-layer keep quotas; this problem is a greedy Fisher-proportional ledger with integer bookkeeping and no comparison of alternatives.
- `kv-quest-page-bound`: an online per-decode-step page budget; this one allocates ranks offline once and they then get baked into the weights.
