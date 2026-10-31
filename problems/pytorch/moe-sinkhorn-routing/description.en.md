# MoE: Sinkhorn Doubly-Stochastic Routing and Column Collapse

## Task

Implement `solve`: turn one batch of router logits into an approximately **doubly stochastic** assignment matrix — every row (token) is normalized over experts, and every column (expert) is iteratively pushed toward load 1. Optionally sharpen the assignment with one element-wise squaring.

This is a restricted numerical model of the "balance experts by a doubly stochastic projection instead of an auxiliary loss" family in MoE research. The iteration count, the alternating direction, where eps sits, the squaring branch and the NaN semantics are all fixed by the statement. It is not a step-by-step reproduction of any training framework, and it involves no gradients or parameter updates.

## Inputs

- `logits`: CPU float64 `[T,E]`, both dimensions non-empty; noncontiguous tensors are accepted and no input is mutated.
- `iters`: a true Python `int` with `0 <= iters <= 16`; `bool`, floats, negatives and out-of-range values are invalid.
- `tau`: a positive finite `int`/`float`, default `1.0`; the softmax temperature.
- `square`: a true Python `bool`, default `False`.
- `eps`: a non-negative finite `int`/`float`, default `0.0`.
- `T,E = 1..8`.

Raise `ValueError` for invalid structure or invalid scalars.

## Exact rules

The starting assignment is a temperature-scaled row softmax:

$$
R^{(0)}_{ij}=\frac{\exp(z_{ij}/\tau)}{\sum_{j'}\exp(z_{ij'}/\tau)}.
$$

Each round performs two normalizations, **columns first, then rows**, and every divisor is "sum plus eps":

$$
C^{(t)}_{j}=\varepsilon+\sum_{i}R^{(t-1)}_{ij},\qquad
R'^{(t)}_{ij}=\frac{R^{(t-1)}_{ij}}{C^{(t)}_{j}},
$$

$$
D^{(t)}_{i}=\varepsilon+\sum_{j}R'^{(t)}_{ij},\qquad
R^{(t)}_{ij}=\frac{R'^{(t)}_{ij}}{D^{(t)}_{i}}.
$$

Column normalization pulls each expert's load toward 1 along the token axis; row normalization makes each token's assignment sum back to 1. A round ends on row normalization, so the returned matrix always has **exact row sums of 1**, while column sums are only approximated: when $T\neq E$ the columns cannot all reach 1, because the total mass is only $T$.

After all `iters` rounds, if `square=True`, sharpen once and renormalize the rows:

$$
Q_{ij}=\frac{\left(R^{(\mathrm{iters})}_{ij}\right)^{2}}{\varepsilon+\sum_{j'}\left(R^{(\mathrm{iters})}_{ij'}\right)^{2}}.
$$

If `square=False`, then $Q=R^{(\mathrm{iters})}$. With `iters=0` and `square=False` the result is an ordinary row softmax.

eps is added to the **divisor**, not to the numerator. With `eps > 0` an all-zero column stays 0 after normalization ($0/(0+\varepsilon)=0$); with `eps = 0` normalizing such a column computes $0/0=$ NaN, and the NaN spreads across the whole row through the row sum. The task requires returning that NaN faithfully, with nothing masked away.

## Return value

Return a two-item `tuple`:

```text
(Q, col_load)
```

| Item | Shape / type | Meaning |
| --- | --- | --- |
| `Q` | float64 `[T,E]` | The normalized assignment matrix |
| `col_load` | float64 `[E]` | Column sums of `Q`, that is each expert's actual load in this batch |

Compare floats with `rtol=2e-11, atol=2e-11`; `NaN` equals `NaN`.

## Example

```python
logits = [[0.0, 0.6931471805599453],
          [0.0, 0.6931471805599453]]

solve(logits, 0)
# Q = [[0.333333, 0.666667],       col_load = [0.666667, 1.333333]
#      [0.333333, 0.666667]]

solve(logits, 1)
# Q = [[0.5, 0.5],                 col_load = [1.0, 1.0]
#      [0.5, 0.5]]

solve(logits, 0, 1.0, True)
# Q = [[0.2, 0.8],                 col_load = [0.4, 1.6]
#      [0.2, 0.8]]
```

`exp(0.6931471805599453) = 2`, so the row softmax is $[1/3,\,2/3]$; both rows are identical, hence column sums $[2/3,\,4/3]$. One column-normalization step divides each column by its own sum, giving $(1/3)/(2/3)=0.5$ and $(2/3)/(4/3)=0.5$, so the matrix becomes all $0.5$. Its row sums are already 1, row normalization changes nothing, and every further round is a fixed point. The squaring branch maps $[1/3,2/3]$ to $[1/9,4/9]$ and dividing by the row sum $5/9$ yields $[0.2,0.8]$.
