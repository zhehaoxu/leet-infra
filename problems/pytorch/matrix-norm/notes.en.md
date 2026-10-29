# Explained: Matrix Norm (fro/nuc and Induced 1/2/inf Norms)

## Origin

`torch.linalg.matrix_norm` is part of the new 1.9 linear-algebra API and its `ord` roster follows the `np.linalg.norm` convention. The closed forms of the induced norms (column sums / row sums) are standard results in Horn & Johnson's "Matrix Analysis"; the spectral and nuclear norms are just the $\ell_\infty$ and $\ell_1$ norms of the singular-value vector. Engineering touch points: spectral normalisation of gradient-reversal layers (Ganin), Spectral Norm Regularization (Yoshida & Miyato 1705.10921) which constrains only $\sigma_{\max}$, and the nuclear norm as a sparsity surrogate in low-rank/distillation work.

## Why the Induced 1 / inf Norms Are "Column Sums / Row Sums"

By definition $\|A\| = \max_{\|v\|_p = 1}\|Av\|_p$. For $p=1$:

$$\max_{\|v\|_1=1}\|Av\|_1 = \max_{\|v\|_1=1}\sum_j\Big|\sum_i a_{ji}v_i\Big| \le \max_i\sum_j|a_{ji}|$$

Choosing $v$ as the unit vector that puts all mass on the most favourable input component $i$ attains the bound, hence $\|A\|_1 = \max_i \sum_j |a_{ji}|$ = **maximum column sum** (the index $i$ runs over columns). Likewise for $p=\infty$, $\|Av\|_\infty \le \max_j\sum_i|a_{ji}|\cdot\|v\|_\infty$ with equality at $v=\mathrm{sign}(A_{j\cdot})$, giving the **maximum row sum**. The two are transposes of each other, $\|A\|_1 = \|A^{\mathsf T}\|_\infty$ — remembering that one identity is enough to keep the direction right.

The reference solution matches this derivation literally:

- `ord=1/-1` → `xd.abs().sum(dim=ra, keepdim=True)` sums along the **row axis** (collapsing rows, keeping the column direction), then `amax/amin` along the **column axis**;
- `ord=inf/-inf` → sum along the **column axis**, then take the extreme along the **row axis**.

## fro / nuc / 2 / -2: Reduction Axes and Batch Layout

`"fro"` is $\sqrt{\sum a^2}$, the same formula as the vector 2-norm. $2/-2/\text{nuc}$ need singular values, and `torch.linalg.svdvals` requires the matrix axes in the **last two dimensions**, so

$$y = \text{movedim}(x; (r, c) \to (n-2, n-1)), \qquad s = \text{svdvals}(y)$$

`svdvals` returns non-negative, descending $s$, so `max`, `min` and `sum` are all unambiguous numbers (there is no singular-vector sign issue here). **The trap is the layout**: after `movedim` the remaining batch axes sit in ascending original order, so `v` no longer has "the same axis layout as $x$ with reduced axes of length 1". The reference restores it with a `_restore` helper:

```
kept = [i for i in range(ndim) if i not in (ra, ca)]
full = [1] * ndim
for i, s in zip(kept, v.shape): full[i] = s   # fill each retained axis
v = v.reshape(full)
```

Reshaping straight to `x`'s shape or tacking on `unsqueeze(-1)` puts the length-1 axes in the wrong places for choices such as `dim=(0,1)` or `dim=(0,2)` — the most common bug in this family. Whenever a reduction over explicit dims has to be broadcast back into the original layout, build the shape axis by axis from the **retained** dimensions.

## Core Points and Pitfalls

1. **Two worlds for `ord=1`**: for a vector it is $\sum|x|$, for a matrix the maximum column sum. Judging the two problems side by side is exactly what exposes the difference.
2. **`dim` must be exactly two ints**: unlike the vector norm, which accepts tuples of any length, `torch.linalg.matrix_norm` takes only a pair.
3. **`-2` is the smallest singular value, not "minus the spectral norm"**; for non-square matrices there are only $\min(m,n)$ values, and `2`/`-2` are its first and last elements.
4. **bool is not int**: `ord=True` would slip into the `== 1` branch, so the reference excludes bool explicitly.
5. **Batch axes survive when `dim` excludes them**: with `x.dim()>2` and `keepdim=False` the output shape is "everything but the two axes", not a scalar.

## How It Differs from Neighbouring Problems

- `vector-norm` (order358, same batch): the sister problem covering the vector-side `ord` conventions and multi-axis reduction.
- `lp-pool-unpool`: $L^p$ pooling is a per-axis statistic with a $1/n$ factor and has no notion of induced norms.
- `rmsnorm` / `layernorm-forward`: affine normalisation forwards that never touch singular values.
- `distributed-clip-grad-norm`: flattens many gradient tensors into one vector and takes a global norm — the vector convention.

## Cross-Checking and Complexity

- The oracle is `torch.linalg.matrix_norm`, over eight `ord` settings × several `dim` choices (adjacent, non-adjacent, negative axes) × both `keepdim` states, with 3-D/4-D batched inputs and illegal-argument probes.
- Complexity: `fro/1/-1/inf/-inf` cost $O(mn)$; `2/-2/nuc` run a full SVD at $O(mn\min(m,n))$. Judging matrices stay $\le 6\times 7$, so float64 remains in the millisecond range.
