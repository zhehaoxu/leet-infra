# Matrix Norm: fro/nuc and Induced 1/2/inf Norms

## Problem Description

Matrix norms come from two unrelated lineages: the **elementwise** Frobenius and nuclear norms (square the entries, or add up singular values) and the **induced** operator norms, where $\max_{\|v\|=1}\|Av\|$ collapses to row/column extremes. `torch.linalg.matrix_norm` drives both from one `ord` argument: `1` is the **maximum column sum** and `inf` the **maximum row sum**, with no relation at all to the vector 1-norm's "sum of magnitudes". The spectral norm $\sigma_{\max}$ and nuclear norm $\sum\sigma_i$ are the max and sum of singular values, showing up in weight regularisation, spectral normalisation (GANs) and low-rank approximation.

$$\|A\|_{\mathrm{fro}} = \Big(\sum_{ij} a_{ij}^2\Big)^{1/2}, \qquad \|A\|_1 = \max_j \sum_i |a_{ij}|, \qquad \|A\|_\infty = \max_i \sum_j |a_{ij}|, \qquad \|A\|_2 = \sigma_{\max}(A), \qquad \|A\|_{\mathrm{nuc}} = \sum_i \sigma_i(A)$$

## Rules

- The full signature and parameter semantics live only in the starter docstring; that is the single source of truth for judging.
- `dim` takes **exactly two ints** (row axis, column axis), default `(-2, -1)`; every other axis is a batch axis and the input must be at least 2-D.
- `ord` accepts only `"fro"` (default), `"nuc"`, `1`, `-1`, `"inf"` / `float("inf")`, `"-inf"`, `2`, `-2`; anything else raises ValueError.
- The negative settings take the **other extreme**: `-1` is the minimum column sum, `-inf` the minimum row sum, `-2` the smallest singular value.
- With `keepdim=True` both reduced axes stay as length 1, **at their original positions** (the batch shape is unchanged).

## Example

$M=\begin{pmatrix}1&-2&3\\4&0&-1\end{pmatrix}$, scalar results for all eight `ord` settings (numbers taken from the EXAMPLE_* constants filled in from test_cases.py):

```
M = [[ 1, -2,  3],
     [ 4,  0, -1]]

fro  = 5.5677642822265625    # sqrt(31)
nuc  = 7.860636711120605
1    = 5.0                   # max of column abs-sums (5, 2, 4)
-1   = 2.0                   # min of the same group
inf  = 6.0                   # max of row abs-sums (6, 5)
-inf = 5.0                   # min of the same group
2    = 4.159660339355469     # largest singular value
-2   = 3.7009761333465576    # smallest singular value
```

The column sums are $\sum_i|M_{i0}|=5$, $\sum_i|M_{i1}|=2$, $\sum_i|M_{i2}|=4$, so `ord=1` gives 5 and `ord=-1` gives 2; the row sums are 6 and 5, so `ord=inf` gives 6.

## Constraints

- Input is float32 (possibly batched), computed in float64, returned as float32; compared against `torch.linalg.matrix_norm` with atol=1e-5, rtol=1e-5
- Singular values are non-negative and descending, so the 2/-2/nuc settings cross-check stably; **singular vectors are never asserted** (their signs are undefined)
- A `dim` that is not a pair, out of range, or with two equal axes, and any `ord` outside the roster, must raise ValueError
