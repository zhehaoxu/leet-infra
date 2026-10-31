# notes — la-gla-chunked (order 659)

## Source

- Official implementation: `fla-org/flash-linear-attention`, `fla/ops/gla/naive.py::naive_recurrent_gla` (`h = h·exp(gk_i)[..., None] + k_i[...,None]·v_i[...,None,:]`, `o_i = (q_i·scale).sum(-2)`, `scale = K**-0.5`, cast to `.float()`). The official chunk side (`fla/ops/gla/chunk.py`) uses BT blocks + matmul rearrangement (with GroupNorm/head splits); this problem only covers the **mathematical identity**: chunked-parallel form ≡ naive recurrence, not kernel tiling details.
- Paper: 2312.06635 (Gated Linear Attention Transformers with Hardware-Efficient Training).
- Boundary with 651 `la-gated-deltanet-recurrent`: that one is the delta rule (subtractive state update `h += k⊗(v − h·k)·β`, g as multiplicative gating); this one is plain GLA (no delta subtraction) and tests the chunk identity itself.

## Judging surface

- solve args: q/k/v/g `[T,·]` (g `[T,K]`), chunk_size, h0 `[K,V]` optional, scale optional (default K^-0.5). Returns o/final_state/chunk_states (state after each chunk, as a ledger).
- All computation in float64 (pinned); oracle `test_cases._py` is an independent pure-Python naive recurrence, tolerance 1e-8/1e-6.
- Anchors: (1) single-chunk K=V=1 hand computation (`o1 = exp(S1−S0)·k0v0 + k1v1`); (2) chunk=1 two chunks (the inter term `(q·exp(g1))·h1` observable); (3) h0 contribution `q·scale·exp(g0)·h0`; (4) random tensors T=7/chunk=3 (uneven tail chunk) vs the naive recurrence; (5) chunk_size ∈ {1,2,4,6,9>T} all agree (identity independent of splitting); (6) default scale vs scale=1 ratio = K^-0.5.
- Invalid inputs: q/k/g shape mismatch, v row count ≠ T, chunk_size ≤0/non-integer/bool, empty input, bad h0 shape, scale ≤0/inf/nan → ValueError. Note `v[:, :1]` is a legitimate V=1 input, **not** invalid.

## Observed pitfalls (real bugs found before mutation testing)

1. **torch broadcasts along the trailing dim**: `torch.exp(gc[-1]) * h` (`[K] × [K,V]`) multiplied the decay into the **columns** (value channels); silently wrong when K=V=2, exposed by the chunk=1 vs naive comparison. Fix: `torch.exp(gc[-1])[:, None] * h` (row-wise per key channel). This mistake is enshrined as the `row-decay-broadcast` mutation.
2. **A-matrix dims**: `g` is `[T,K]`; careless `exp(gc)[:, None, :]` bookkeeping can turn `[l,K]` products into `[l,l,K]`; the correct formulation decays each side per token first (`qc ⊙ exp(gc)` and `kc ⊙ exp(−gc)`) then does `[l,K]@[K,l]`.
3. Anchor tensors must be constructed with `dtype=torch.float64`: default float32 rounds constants like `-ln2` first, blowing up 1e-12-level asserts.

## Mutation log (9, all caught)

inter-no-decay, intra-no-mask (future leak), intra-diag-drop, state-no-end-decay, state-k-no-decay, dec-end-from-zero (state off by a factor `exp(S_end)`), scale-drop, cumsum-suffix, row-decay-broadcast (the real bug above).

## Red lines

No head splits/GroupNorm/kernel tiling (BT matmul rearrangement, inter/intra fusion); no batch/head dims (single-sequence single-head convention); dedup against `cumsum-scan` and the attention family.
