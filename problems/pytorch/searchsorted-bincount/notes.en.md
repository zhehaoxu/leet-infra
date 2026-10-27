# In Depth: searchsorted Bucketing and bincount Histogram

## Source

- Operators being replicated: `torch.searchsorted(boundaries, x, right=False)` (requires `boundaries` sorted ascending along the last dimension, binary search per element) and `torch.bincount(input, weights=None, minlength=0)` (requires a 1-D non-negative integer tensor). This task fuses the two calls and **pins** `counts` to length \(M+1\) — equivalent to always passing `minlength=M+1` to the native call. Semantics follow the PyTorch official docs plus measurements taken in this repository.
- Where the pair shows up (all of them are "look up a bin, then count bins"):
  - inverse-CDF sampling: in `top-p-sampling` and `rl-per-priority-replay` the rule "smallest index whose prefix sum is **strictly greater** than \(u\)" *is* the `right=True` convention (those notes already confirmed it equals `searchsorted(cdf, u, right=True)`);
  - MoE load statistics: `moe-switch-aux-loss` counts expert selections with `bincount(selected.reshape(-1), minlength=E)` — the `minlength=E` plays exactly the same role as "an empty trailing bin still occupies a slot" here;
  - codebook encoding: `quant-nf4-quantile-grid`'s 16-segment quantile codebook plus nearest code is "look up the segment by boundary", written there as an explicit \(\arg\min_k|u-T_k|\).
- CPU, fully deterministic: lookup uses only comparisons, counting only accumulations.

## Full derivation and index convention

**Why the counting rule equals the binary-search answer**: let `boundaries` be ascending with length \(M\), and write \(k(x)=\#\{j: b_j < x\}\). By sortedness the set of indices below \(x\) is exactly the prefix \(\{0,\dots,k-1\}\), hence
\[b_{k-1} < x \le b_k \quad(\text{with the convention } b_{-1}=-\infty,\ b_M=+\infty),\]
so \(k\) *is* "the smallest \(i\) with \(b_i \ge x\)" — the definition of `side="left"`. Likewise \(\#\{j: b_j \le x\}\) is "the smallest \(i\) with \(b_i > x\)", i.e. `side="right"`. The reference solution therefore just computes
\[\text{idx} = \sum_j [\,b_j < x\,]\quad\text{or}\quad \sum_j [\,b_j \le x\,]\]
(broadcast to an (N, M) boolean matrix, summed over M) and never calls a binary search, yet matches the native result exactly — this is the structurally independent solution path.

**The closure direction flips with `right`**: from the above, with `right=False` \(x\in(b_{i-1},\,b_i]\) (right-closed) and with `right=True` \(x\in[b_{i-1},\,b_i)\) (left-closed). The "left-closed, right-open" sentence in the starter docstring is the `right=True` reading; the counting definition governs. **Minimal discriminator**: with \(b=[1,3,3,7]\), \(x=[0,3,5,9]\) only \(x=3\) sits on the repeated run and the branches give 1 versus 3; \(x=5,9,0\) are far from any boundary and agree.

**Duplicate boundaries create identically empty bins**: if \(b_{i-1}=b_i=c\), bin \(i\)'s interval is empty under both conventions (\((c,c]\) right-closed, \([c,c)\) left-closed), so `counts[i]` is always 0. The repeated run \([3,3]\) in the example produces **bin 2, empty in both branches**. What does move with `right` is which slot a value sitting exactly on a boundary lands in: `x=3` goes to bin 1 (\((1,3]\)) under `left` and bin 3 (\([3,7)\)) under `right`, so `counts[1]` changes 1→0 and `counts[3]` changes 1→2.

**`counts` length must be \(M+1\)**: native `torch.bincount(idx)` returns length \(\max(\text{idx})+1\) (measured here: \(b=[1,3,7]\), \(x=[0,2]\) gives idx \([0,1]\), a bare bincount returns the length-2 `[1,1]`, and only `minlength=4` yields `[1,1,0,0]`). The task pins the length to \(M+1\), so an implementation must either pad explicitly or, like the reference, allocate `torch.zeros(m+1)` first and then `index_add_`.

**bincount is a special case of scatter_reduce**: `counts[i] = Σ_{idx==i} 1` (or of the weights) is exactly a one-dimensional `scatter_reduce` with `reduce="sum"`, `src ≡ 1` (or src = weights) and `include_self=False`; the relation to the same-batch `scatter-reduce-aggregate` is "special case vs full generalisation". `index_add_` is the compact equivalent and keeps untouched slots at 0.

**Three dtype conventions**: `idx` is always int64 (like native `searchsorted`); unweighted `counts` is always int64; weighted `counts` follows the dtype of `weights`, and judging only feeds floating-point weights. Note that native `torch.bincount` with **integer** `weights` silently promotes to **float64** (measured: `bincount([0,1,1], weights=[1,2,3])` → `tensor([1., 5.], dtype=torch.float64)`), which differs from this task's "dtype follows weights"; the reference casts integer weights to float32. Judging does not cover that branch, so copying the native behaviour would pass while still contradicting the docstring.

**Two degenerate empties**: with \(M=0\) both counting rules give 0, so `idx` is all zeros and `counts` has length 1 holding `x.numel()` (or the weight sum) — "the only bin is the whole real line". With an empty `x`, `idx` is empty and `counts` is an all-zero vector of length \(M+1\); a multi-dimensional empty input (e.g. shape (0,3)) still returns `idx.shape == x.shape`. Neither should raise.

**Error surface**: `boundaries.dim() != 1` raises RuntimeError (matching native `searchsorted`); a `weights` shape mismatching `x` raises ValueError (this task's own argument check). The split is deliberate — don't unify it.

## Key points and common mistakes

1. **Hard-code the length \(M+1\)**: a bare `torch.bincount(idx)` comes out short whenever the trailing bins are empty (measured: length 2 for idx \([0,1]\)); judging builds expectations as `bincount(..., minlength=M+1)` and compares with `torch.equal`, so a length mismatch fails outright, and the empty-`x` / empty-`boundaries` cases pin both all-empty shapes.
2. **`right` only changes values equal to a boundary**: implementing it as "shift everything one slot right" is wrong — elements far from boundaries agree in both branches.
3. **Empty bins and moving slots are different things**: the repeated run \([3,3]\) makes bin 2 identically 0 under both settings; what moves is the on-boundary value itself (`counts[1]` 1→0, `counts[3]` 1→2 in the example).
4. **The bin index is a count, not "the index of the boundary hit"**: below all boundaries it is 0, at or above the last one it is \(M\); the range \([0,M]\) has \(M+1\) values.
5. **For multi-dimensional `x`, `idx` keeps `x.shape` and counting flattens**: `counts.sum()` always equals the element count regardless of shape.
6. **Weighted `counts` dtype follows `weights`**: float64 weights must produce float64 output; accumulating in float32 and casting up afterwards diverges (judging compares against native `bincount(weights=...)` with \(10^{-6}\)).
7. **Unweighted `counts` must be integral**: a float accumulator cast at the end loses exactness at large `numel`; judging asserts with `torch.equal`.
8. **Do not sort unsorted boundaries**: the contract guarantees ascending order; sorting would break the "bin \(i\) ↔ interval" binding (bin numbers are tied to boundary values), and the reference's broadcast-compare path would give yet another answer — judging never feeds unsorted input, so don't add logic for it.
9. **Integer weights are not floating weights**: the docstring says floating tensor; judging only feeds float32/float64.
10. **Empty `x` and empty `boundaries` are both legal**, with predictable shapes (`idx.shape == x.shape` and `counts.shape == (1,)` respectively).

## Differences from neighboring problems

- **vs `scatter-reduce-aggregate` (same batch)**: that one has five reductions, any number of dimensions and both `include_self` states; this is the one-dimensional, always-`sum`, never-include-self special case, and the extra half of the work here is the lookup.
- **vs `gather-scatter`**: there the index tensor is *given* and drives reads/overwrites; here the indices are *computed* from numeric comparisons — a "value → index" mapping.
- **vs `sort-argsort`**: ranks after sorting can also express bucketing, but that is a global \(O(M\log M)\) operation; boundaries here are already ascending, only a per-element search/compare is needed, and there is no stability-of-ranking notion to preserve.
- **vs `cumsum-scan`**: prefix-sum tasks often use `searchsorted` downstream (inverse lookup on a CDF); this problem provides no cumsum and does not assume the boundaries are a cumulative sum.
- **vs `moe-switch-aux-loss`**: that task only counts *integer* labels (bin numbers are inputs); this adds the whole lookup step and the `right` convention.
- **vs `grid-sample` / `upsample-bilinear`**: those also map coordinates to indices, but via an affine coordinate transform with bilinear weights — no ordered boundaries and no closure direction.

## Cross-check convention and complexity

- **Primary oracle**: `torch.searchsorted(b, x, right=...)` plus `torch.bincount(idx, minlength=M+1)` (weighted: `bincount(idx, weights=w, minlength=M+1)`), over 5 random (M=7, N=40) draws × both `right` values × weighted/unweighted, a multi-dimensional `x` of shape (5,8), an int64 group and a float64 group; indices and integer counts use `torch.equal`, floating counts \(10^{-6}\).
- **Second oracle**: `_oracle_scan` walks boundaries with a pure-Python `while bl[i] < v: i += 1` **linear scan** for the bin number, then uses `list.count` / a per-bin `sum` for the counts — different from both the solution ("broadcast compare + index_add_") and the native binary search, which is what catches off-by-one (`<` written as `<=`) and a missing trailing bin. Its cases include duplicate boundaries (`[2,2,2,5,5]`), empty boundaries (float and int), and a weighted (3,4) two-dimensional `x`.
- Discriminating cases: the two-branch example table (`idx_left/idx_right`, `counts_left/counts_right`, all filled from measured output), explicit assertions on duplicate-boundary indices, endpoint-equal cases where the branches differ by exactly 1, both out-of-range sides (\(-5,0,100\)), empty `x` (2-D empty and weighted empty), empty boundaries (weighted and both-empty), the weighted/unweighted dtype and sum identities, float64.
- Complexity: the broadcast-compare solution is \(O(N\cdot M)\) time with an \(O(N\cdot M)\) temporary boolean matrix; native binary search is \(O(N\log M)\) with smaller constants. The counting step is \(O(N)\) (`index_add_` or `bincount`) and writes \(M+1\) slots. The implementation path is unconstrained, and at judging sizes (\(N\le 40\), \(M\le 9\)) both are far inside the tolerance budget.
