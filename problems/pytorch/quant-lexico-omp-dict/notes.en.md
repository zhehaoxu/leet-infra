# Solution notes · Lexico's row-wise OMP and CSR packing

## One-liner

Treat every KV vector as a signal and greedily pick `n_atoms` atoms from a fixed dictionary with OMP: each round first tests the relative-energy stopping condition, then takes the unselected atom maximising $|\langle r,a_j\rangle|$, then refits least squares over the whole selected set; finally flatten each row's "atom ids + weights" into a CSR triple.

## How upstream actually runs

`lexico/omp.py::omp_v0` (L10-92) is a batched classical OMP, but it has no explicit atom mask and never really inverts anything:

- L28 computes `projections = bmm(Xᵀ, yᵀ)ᵀ` once, then L84 downdates it each round with `projections -= temp_a_F * D_mybest[k]`, and the `rsqrt(1 - innerp(...))` at L74 involves the new atom's Gram components against the older ones — the `F`/`a_F` machinery (L31-32, L87-88, and `F.permute(...) @ a_F` at L90) is a Cholesky-style incremental version of $(X_S^\top X_S)^{-1}X_S^\top y$, mathematically equivalent to "refit the whole set", which is the plain formulation the statement uses.
- The selection step, L64, is `sets[k] = projections.abs().argmax(2)`: an absolute value, `argmax` returning the first maximum (the smallest index), and **no exclusion of already-selected atoms** — repetition is avoided only because their correlations have been downdated to 0.
- The stopping step: the loop at L43 is `range(n_nonzero_coefs + (tol is not None))`, L46 tests `normr2 <= tol` **before** the pick, L41 pre-multiplies the threshold into `normr2_init * tol * tol`, and L54 records `result_lengths = k`. So "test before picking, `lengths` counts what was already picked, error reported at the stopping point" is upstream verbatim.
- `omp()` (L94-112) does the packing: L101-104 falls back to the full budget when `tol` is absent, L106 casts weights to `float8_e4m3fn`, L107 casts atom ids to int16, and L109-110 builds `indptr` as an int32 prefix sum.
- `cache_utils.py::_compress` (L159-191) feeds `tensor.permute(2, 0, 1, 3).reshape(1, -1, head_dim)` (L163) to `omp`, i.e. head × token flattened into rows; `CHUNK_SIZE = 20000` (L160) only accumulates `indptr` across chunks; K and V each have their own per-layer dictionary (L89-90, L136-137). `max_sparsity` defaults to 14 (L14) and `error_threshold` to `None` (L15) — the repository's `examples/generate.py` L20 uses 8 while `examples/lm_eval_harness.py` L67 uses 14.

Provenance: `README.md` L7 (badge) and L76 (BibTeX entry) both give arXiv 2412.08890; the dictionary is trained offline on WikiText-103, has roughly 4k atoms and is reused across tasks and models (L15, L21-22).

## Five places where this problem departs from upstream (all pinned in the statement)

1. **`tol=0.0`**: upstream's L37 is `if tol:`, so $0.0$ is falsy and silently means "early stop disabled"; this problem tests `tol is not None`, making $0.0$ a live zero threshold — a zero row stops at `lengths=0` in round 0.
2. **Explicit exclusion of selected atoms**: upstream masks nothing, so a degenerate tie (residual exactly zero) can pick the same column twice; here only unselected atoms are eligible.
3. **Dtypes**: upstream uses `int16 / float8_e4m3fn / int32`; this problem uses int64 plus the input dtype and applies no low-bit truncation.
4. **How least squares is solved**: upstream uses the incremental Gram factor, this problem uses `lstsq`; the two agree on full-rank sets, and on a rank-deficient set this problem takes the **minimum-norm solution** (the branch `torch.linalg.lstsq` and `numpy.linalg.lstsq` share) while `recon` remains the unique projection.
5. **Argument layout**: upstream is a batched $(1,N,D)$ call with chunk concatenation; here rows are given directly as $(N,D)$ and the dictionary as $(D,M)$, with `CHUNK_SIZE` and float8 storage left out of the problem entirely.

## Grading points

All six keys must match in dtype and shape: `lengths/csr_indptr/csr_indices` int64, `csr_values/residual_norm/recon` following the input dtype, and both `csr_values` and `csr_indices` have length `lengths.sum()`. The selection order is "the order atoms were picked", not sorted indices — slicing a CSR row, scaling by its weights and summing the atoms must reproduce that row's `recon`, which is exactly how the order is checked. Zero rows, `n_atoms == M`, and scaling a whole row by 10 (order unchanged, weights scale too) each have dedicated assertions.

## Edges

With `tol=None` every row always fills the budget; with a very loose `tol` (say 0.99) a row may select nothing at all in round 0 — possible only when $I = 0$. $D$ carries no divisibility requirement and $M$ only needs to be at least `n_atoms`. Dictionary columns need not be normalised and need not be independent — `check_selection_is_abs_correlation_with_ties` deliberately feeds three duplicate columns.

## Complexity

Per row and round: one $O(DM)$ correlation sweep and one $O(D\,k^2)$ least squares, totalling $O(N\cdot \text{n\_atoms}\cdot(DM + D\,\text{n\_atoms}^2))$; the dictionary is an input, so none of the K-SVD training cost is involved.

## Related problems

`spqr-sparse-outlier` is one-dimensional sparsity ("store the outliers separately"); this problem is multi-atom linear combination over a dictionary. `quant-squeezellm-sensitive-kmeans` represents a span with a single codeword, whereas here $k$ atoms are combined with weights. `retrieval-pq-codes` shares the codebook family but has no per-row greedy search.
