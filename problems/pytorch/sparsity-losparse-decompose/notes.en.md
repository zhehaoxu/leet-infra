# Solution Notes: LoSparse Low-Rank + Structured Sparse Decomposition Step

## Source Localization

- Source: yxli2123/LoSparse (the official ICML 2023 repo; compressics/ and IntelLabs/ do not exist), `utils.py`, 399 lines. The training-side PLATON pruner (ipt/exp_avg_unc and the structured_prune mean/sum/max/min row-column aggregations) is training-convergence machinery — red-lined and excluded; only the static decomposition and forward are taken.
- `low_rank_decomposition` (L7-66): svd(full_matrices=False) → rank=count_nonzero(S) → parameter_ratio first with ceil(pr·HW/(H+W)), else ceil(rank·rank_ratio) → the √diag(S) split into L/R, max takes `[:, :k]`, min takes `[:, -k:]`, random uses random.choices (randomness red line, excluded).
- `substitute_layer_weights` (L183-194): `S = W − L@R` becomes the sparse residual inside LinearLoSparse.
- `prune_sparse` (L136-139): `nonzero_idx = nonzero(sparse.weight.sum(dim=1))` (out-row sums) → `sparse_weight_pruned = S[nonzero_idx]`.
- The two forward paths (L91-126): unpruned `LRX + sparse(x)`; pruned `x @ pruned.T` scattered into `[..., nonzero_idx]` (the SX cache is Deberta's double-Q/K engineering, out).
- Boundary with `quant-gear-lowrank-outlier`: GEAR is "SVD low rank + residual magnitude top-k element sparsity" (unstructured, quantization side); LoSparse is "low rank + **row-structured** pruning + joint fine-tuning during training" — this problem tests the budget formula/√ split/row-prune scatter, no mechanism overlap.

## Math Notes

1. **The √diag split**: W ≈ U_k diag(s_k) V_k^T = (U_k diag(√s)) @ (diag(√s) V_k^T) = L@R with balanced parameters (column/row norms both √s_i); the drop-√ mutant is caught by both the column-norm assertion and the oracle.
2. **The parameter budget formula**: k = pr·HW/(H+W) makes (H+W)·k ≈ pr·HW — the low-rank branch's parameter share. A positive ratio through ceil is always ≥1, so k<1 is dead code; only the upper bound k>min(H,W) is checked.
3. **Rank-deficiency counting**: `count_nonzero(S)` is fragile on float32 SVD — duplicated rows leave ~1e-8 nonzero singular values and still count as full rank; only **exactly zero rows** (trailing rows zeroed) produce exact 0 singular values. Tests must use zero rows, not duplicated rows.
4. **Automatic drop-out in row pruning**: a masked row is all-zero → row sum 0 → excluded by nonzero → `S_masked[idx]` equals `S_res[idx]` elementwise (the basis of the equivalent mutant m-pruned-nomask); `abs().sum` vs `sum` is exactly equivalent for nonzero selection (|s|≠0 ⟺ s≠0, equivalent mutant m-nonzero-abs).
5. **The scatter-forward identity**: `scatter(x@S_pruned.T, idx) == (x @ (S·mask).T)` — pruning is an implementation detail of sparsity, not a change of math.

## Implementation Notes

- Validation order: weight 2-D → x 3-D with in-dim = W → criteria in the pair → parameter_ratio positive or None → rank_ratio positive → neuron_mask 1-D of length H and 0/1 → bias 1-D of length H or None → decompose → k upper bound → all-masked check.
- The decomposition uses float32 SVD (following the official code, no double promotion); the oracle shares svd as a primitive (the same layered pattern as 770).
- The scatter zeros use x.dtype; `SX_scattered[..., nonzero_idx] = x @ sparse_pruned.T`.

## Testing Notes

- Oracle-mirrored bookkeeping + shared svd; L/R/residual/nonzero/forward all compared.
- Reconstruction properties: L@R == the first-k (or last-k) SVD triplet reconstruction; L's column norms == √s_i (the √-split-specific assertion).
- Criteria contrast: min vs max give different L; the rank_ratio path must use a **rank-deficient weight** (rank < min(H,W)) to distinguish `rank·ratio` from `min(H,W)·ratio`.
- The pruning identity: pruned forward == full-residual forward + mask zeroing (scatter semantics); nonzero_idx == nonzero((residual·mask).row sums).
- The unpruned forward identity: Y == x @ weight.T (the decomposition is an exact linear split W = LR + S).
- Rank deficiency: zero-row construction (exact 0 singular values) → rank/is_full_rank.
- Validation battery: 2-D/3-D/in-dim/random/ratio 0 and bool/mask length and 0-1/bias length/k upper bound (pr=5)/all-zero mask.

## Mutation List (35, all CAUGHT; 2 equivalents excluded)

Budget 7: parameter_ratio override flipped/None inverted/ceil→floor/formula numerator-denominator swapped/rank_ratio using min(H,W)/rank counting full rank/is_full_rank always true. SVD 6: √ dropped, L/R split swapped, max↔min, min slicing the head, k−1, Vh transposed. Residual 3: sign flipped, full-SVD residual, residual=W. Pruning 5: mask not multiplied/mask by columns, nonzero by columns/abs (equivalent, excluded). Forward 6: LR order/factor swap, scatter replaced by full/no transpose, unpruned residual term replaced by R.T, bias dropped. Validation 13: weight/x dims, in-dim, criteria, positive ratios, mask 0-1/length, bias, all-masked, message.

Equivalent excluded (2): `m-nonzero-abs` (|s|≠0 ⟺ s≠0, the nonzero selection unchanged); `m-pruned-nomask` (masked rows are all-zero → row sum 0 → auto drop-out, so taking surviving rows from the raw residual is elementwise equal).

Experience: ① **rank-deficiency construction for count_nonzero must use exactly zero rows** — duplicated rows leave ~1e-8 residual singular values under float32 SVD and still count as full rank; ② **never drop an if's colon/closing paren in a replacement string** — a syntax error makes the judge return 0/0, which the `passed < total` survival check misreads as SURVIVED (add a prior check to the mutation script: the replaced source must compile); ③ the rank-budget branch's mutants only diverge with rank < min(H,W) weights (on full-rank inputs `rank·ratio` and `min(H,W)·ratio` agree); ④ nonzero-style assertions are naturally immune to abs/sign mutants (the nonzero selection is unchanged) — record such mutants as equivalents instead of forcing cases.
