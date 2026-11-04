# Sparsity: LoSparse Low-Rank + Structured Sparse Decomposition Step (SVD Factor Split + Parameter Budget + Neuron-Row Pruning)

## Problem

LoSparse approximates a weight matrix by the sum of a low-rank matrix and a structured sparse one: the SVD low-rank part compresses the coherent components across neurons, while structured (neuron/row-level) pruning removes the incoherent ones — the low rank keeps pruning from losing expressive neurons, and pruning enhances the low rank's diversity. Implement `solve(weight, x, parameter_ratio=0.15, rank_ratio=0.1, remove_criteria="max_eigenvalue", neuron_mask=None, bias=None)`:

```text
# 1. Decomposition (official low_rank_decomposition)
U, S, Vh = svd(W, full_matrices=False)      # singular values descending
rank = count_nonzero(S)
parameter_ratio not None: k = ceil(pr · H·W / (H + W))   # parameter budget, overrides rank_ratio
otherwise:                k = ceil(rank · rank_ratio)    # rank budget
max_eigenvalue:   L = U @ √diag(S)[:, :k],  R = √diag(S)[:k, :] @ Vh
min_eigenvalue:   L = U @ √diag(S)[:, -k:], R = √diag(S)[-k:, :] @ Vh
(the √diag split balances L/R parameter counts; L @ R reconstructs the chosen triplet)

# 2. Residual and pruning
S_res = W − L @ R                           # sparse residual
S_masked = S_res · neuron_mask[:, None]     # simulates post-training row zeroing
nonzero_idx = nonzero(S_masked.sum(dim=1))  # surviving neurons with nonzero row sum
sparse_pruned = S_masked[nonzero_idx, :]

# 3. Forward (x is (B, L, in))
unpruned: Y = x @ R.T @ L.T + x @ S_res.T + bias
pruned:   Y = x @ R.T @ L.T + scatter(x @ sparse_pruned.T, nonzero_idx) + bias
```

Return `dict(L, R, sparse_residual, reduced_rank, rank, is_full_rank, nonzero_idx, sparse_pruned, output)`.

## Key Semantics

- **parameter_ratio overrides rank_ratio**: when not None, the parameter budget `ceil(pr·H·W/(H+W))` applies (k implicitly < min(H,W)); only when None does `ceil(rank·rank_ratio)` apply — rank uses `count_nonzero(S)` (exactly-zero singular values of rank-deficient weights are what count).
- **The √diag split is the balancing factor**: L's column norms = R's row norms = √s_i, and L@R is exactly the chosen triplet's reconstruction; dropping √ pushes the squared spectrum into the factors.
- **min_eigenvalue takes the tail**: with descending SVD, `[-k:]` is the smallest triplet (distinct from the max path).
- **Pruning is row-structured**: survival is decided by `sum(dim=1)` (the output-neuron row sum); masked rows are all-zero with sum 0 and drop out automatically — taking surviving rows from the raw residual equals taking them from the masked matrix.
- **Scatter forward**: after pruning only surviving rows are computed, then scattered back by `nonzero_idx` — numerically equal to "full-residual forward + mask zeroing".
- **The official `random` criteria hits the randomness red line and is excluded**; the official `raise NameError` becomes a judgement-domain ValueError.

## Judgement-domain Conventions

- The official `masked_fill_` comes from PLATON training; the judgement domain simulates it with an explicit `neuron_mask` (0/1 row mask), and the residual itself is not trained.
- Weights follow the (out, in) row-major nn.Linear convention, x is (B, L, in); bias is added as-is.
- Official logging and the SX training cache (Deberta's double Q/K use) are out.
- The official code validates nothing; the judgement domain adds weight/x shapes/criteria/positive ratios/mask 0-1 and length/bias length/k upper bound/all-masked checks.

## Example

With H=12, W=8, pr=0.15: k = ceil(0.15·96/20) = 1; L (12,1), R (1,8), L@R = the largest-triplet reconstruction; residual (12,8). Under the mask, surviving rows are decided by row sums and the forward scatters back to (B,L,12). For a rank-deficient weight (trailing rows exactly zero) rank=2 and is_full_rank=False.

Data scale: H ≤ 32, W ≤ 32, x (B ≤ 4, L ≤ 16), float32.
