# Quantization: Lexico Sparse Dictionary Coding of the KV Cache (Row-wise OMP)

## Problem description

Lexico (arXiv 2412.08890) treats the KV cache as a signal rather than as numbers: a small input-agnostic dictionary (roughly 4k atoms) is trained offline on WikiText, and every KV vector is then expressed as a linear combination of **a few atoms** from that dictionary, so what gets stored is only the sparse triple of "atom ids + weights" in CSR form. This problem implements the deterministic half of that pipeline — row-wise orthogonal matching pursuit (OMP) plus the CSR packing.

Dictionary training (K-SVD), the split into chunks of 20000 rows, and squeezing the weights into float8 all belong to the calibration or storage side and are out of scope: the dictionary arrives as an explicit argument and weights are returned in the input dtype.

## Rules

Write `vectors` as $(N,D)$ (each row is one KV vector $y$ to encode) and `dictionary` as $(D,M)$ — **atoms are the columns**, atom $m$ is `dictionary[:, m]`. Rows are processed independently of one another; internals are float64, returned tensors use the input dtype.

Per row:

1. **Initialise**: selected set $S=\varnothing$, residual $r=y$, and initial energy $I=\lVert y\rVert^2$.
2. **Stop-check first**: in round $k$ ($k=0,\dots,$`n_atoms`$-1$), if `tol` was given and $\lVert r\rVert^2 \le \text{tol}^2\cdot I$, the row **ends here**: `lengths` $=k$ (how many atoms were already selected) and no further atom is chosen in this or any later round; if the budget runs out, `lengths` $=$ `n_atoms`.
3. **Pick an atom**: among the atoms **not yet selected**, compute $|\langle r, a_j\rangle|$ and take the largest; ties go to the **smallest index**. Append $j$ to the end of $S$ (the in-row order is the order of selection).
4. **Refit the whole set**: solve least squares over all of $S$ again, $w=(X_S^\top X_S)^{-1}X_S^\top y$, then $r \leftarrow y - X_S w$. **Not** an update of the new atom alone and not an incremental projection — every round that adds an atom rewrites the older atoms' weights.
5. **A zero row is deterministic**: with $y=0$ every correlation is 0, so any `tol` (even `tol=0.0`) makes the round-0 test $0\le 0$ stop immediately with `lengths` $=0$; without `tol` the atoms $0,1,\dots,$`n_atoms`$-1$ are chosen in turn with all-zero weights.
6. **Rank-deficient sets take the minimum-norm solution**: if the selected atoms are linearly dependent (possible when the dictionary carries redundant columns), $(X_S^\top X_S)$ is not invertible and this problem defines the answer as the **minimum-norm least-squares solution**, i.e. what `torch.linalg.lstsq` / `numpy.linalg.lstsq` return; `recon` on that row is unaffected (the projection is unique), only how `csr_values` splits changes.

Return a dict with 6 keys (the CSR convention matches upstream `omp()`):

- `lengths`: int64 $(N,)$, atoms actually used per row;
- `csr_indptr`: int64 $(N+1,)$, the prefix sum of `[0] + lengths`;
- `csr_indices`: int64 $(\text{total},)$, the selected atom ids concatenated row by row;
- `csr_values`: input dtype $(\text{total},)$, the weights matching `csr_indices` one-to-one;
- `residual_norm`: input dtype $(N,)$, $\lVert y-\text{recon}\rVert$ at **that row's own stopping point**;
- `recon`: input dtype $(N,D)$, $X_S w$; rows with `lengths == 0` are all zeros.

## Task

Implement `solve(vectors, dictionary, n_atoms, tol=None)`:

- `vectors`: 2-D float32/float64 $(N,D)$; `dictionary`: $(D,M)$ with the same dtype as `vectors`; both read-only;
- `n_atoms`: a positive integer $\le M$ (Lexico's `max_sparsity`, which defaults to 14 upstream);
- `tol`: `None`, or a float in $[0,1)$.

Non-2-D input, a dtype other than float32/float64, a dtype mismatch between the two tensors, a `dictionary` whose row count is not $D$, an illegal `n_atoms` (`bool` and `3.0` both count) or one larger than $M$, and a `tol` outside $[0,1)$ (`1.0`, `-0.1`, `True` are all rejected) raise `ValueError`.

## Example

$D=3$, $M=5$ with atoms $a_0=(1,0,0)$, $a_1=(0,2,0)$, $a_2=(0,0,1)$, $a_3=(1,1,0)$, $a_4=(0,1,1)$ and `n_atoms=2`. On row 0, $\langle y,a_3\rangle=3$ wins the first round and leaves residual $(0.5,-0.5,0)$; in round two $a_1$ prevails with $|{-1}|$, and the refit gives $w=(2,-0.5)$, driving the residual to zero. On row 1 the first round is a tie between $a_2$ and $a_4$, so the smaller index wins, the residual is already zero, and round two only adds a zero-weight atom. On row 2 the residual after $a_1$ is $(1,0,1)$, every one of $a_0,a_2,a_3,a_4$ correlates by exactly 1, the tie again picks $a_0$, and the budget runs out with 1.0 still unexplained.

```
vectors    = [[2.0, 1.0, 0.0], [0.0, 0.0, 3.0], [1.0, 1.0, 1.0]]
n_atoms    = 2          tol = None

lengths        = [2, 2, 2]
csr_indptr     = [0, 2, 4, 6]
csr_indices    = [3, 1,  2, 0,  1, 0]
csr_values     = [2.0, -0.5,  3.0, -0.0,  0.5, 1.0]
residual_norm  = [0.0, 0.0, 1.0]
recon          = [[2.0, 1.0, 0.0], [0.0, 0.0, 3.0], [1.0, 1.0, 0.0]]
```

With `tol=0.9` all three rows stop in round two ($\lVert r\rVert^2$ is $0.5,0,2$ respectively, none above $0.81\cdot I$), so `lengths` becomes $[1,1,1]$ and `residual_norm` becomes $[0.707107, 0.0, 1.414214]$.

## Constraints

- **The stop test comes before the pick**: `lengths` is how many atoms were already selected at the moment of stopping and `residual_norm` is reported at that same moment; checking after the pick gives every early-stopped row one atom too many.
- **Refit the whole set every round**: the first atom's weight on row 0 moves from $1.5$ (alone) to $2.0$ (with the second atom). A greedy scheme that only appends the new coefficient — the other branch of matching pursuit — is wrong here.
- **The correlation is an absolute value**: $\langle y,a_1\rangle=-8$ still beats $+4$, and weights may be negative.
- **Ties go to the smallest index**: upstream's `argmax` returns the first maximum; do not let floating-point noise choose.
- **`tol=0.0` is a live threshold**: this problem tests `tol is not None`, so a zero row with `tol=0.0` stops immediately at `lengths=0`. Upstream writes `if tol:`, which reads $0.0$ as falsy and treats it as "disabled" — this problem does not copy that.
- **Selected atoms leave the race**: the residual is orthogonal to the span already fitted, and upstream reaches the same effect by downdating the projections without explicitly masking, so a degenerate tie can make it re-pick the same column; here the exclusion is explicit.
