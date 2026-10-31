# Quantization: E8 Lattice Nearest-Point Rounding

## Problem

Scalar quantization rounds each coordinate independently, while vector quantization treats 8 coordinates as one unit and rounds on a higher-dimensional lattice — at the same average bit width, lattice points cover space more efficiently. E8 is one of the densest lattices in 8 dimensions. QuIP# (arXiv:2402.04396) builds its E8P codebook on E8: 227 codewords from `D8+1/2` inside a ball, plus 29 norm-12 codewords, expanded to `2^16` points by sign flips and `±1/4` shifts.

This problem implements only the core **nearest-lattice-point rounding**:

\[
E8 = D8 \cup (D8 + \tfrac12),\qquad
D8 = \{y\in\mathbb{Z}^8 : \textstyle\sum_i y_i \text{ is even}\}
\]

For each 8-D point `x`:

1. **D8 candidate**: `y = round(x)` (coordinate-wise round-half-to-even). If `sum(y)` is even, `y` is already in D8; otherwise exactly one coordinate must move: pick the coordinate with the largest `|x_i - y_i|` and move it by 1 toward `x`, producing an even coordinate sum.
2. **D8+1/2 candidate**: let `z = x - 0.5`, apply the same D8 rounding to `z` to get `y'`, and the candidate is `y' + 0.5`.
3. Compare the squared errors of the two candidates; choose coset 1 only if it is strictly smaller, ties go to coset 0.

## Task

Implement `solve(X)`:

- `X`: float32 `(N, 8)`, non-empty and finite;
- Return a `dict`:
  - `rounded` float32 `(N, 8)`: the nearest E8 point;
  - `coset` int64 `(N,)`: `0` for D8, `1` for `D8+1/2`;
  - `sse` float32 `(N,)`: squared error `||x - rounded||²` per point.

**Conventions**: float64 internally, float32 output, tolerance `1e-5`. Coordinate rounding follows `torch.round` (round-half-to-even). When several coordinates tie for the largest `|x_i-y_i|`, pick the **smallest index**; when the difference is 0, the direction is `+1`. If `X` is not 2-D floating, is not `[N,8]` with `N>0`, or is non-finite, raise `ValueError`.

## Example

```
x = [0.9, 0.6, 0.6, 0.1, 0.1, 0.1, 0.1, 0.1]
round(x) = [1, 1, 1, 0, 0, 0, 0, 0], sum = 3 (odd)
largest |diff| is 0.4 at indices 1 and 2 -> take index 1, direction -1 -> y0 = [1, 0, 1, 0, 0, 0, 0, 0]
sse0 = 0.01 + 0.36 + 0.16 + 5*0.01 = 0.58
z = x - 0.5 = [0.4, 0.1, 0.1, -0.4, ...], rounds to all 0 with even sum -> y1 = [0.5]*8
sse1 = 0.16 + 2*0.01 + 5*0.16 = 0.98
coset0 wins: rounded = [1, 0, 1, 0, 0, 0, 0, 0], coset = 0, sse = 0.58
```

## Constraints

- Both cosets must be tried; D8 rounding alone is not enough.
- The parity test uses the **integer candidate** (for coset 1, the coordinate sum of `y'`, without `+0.5`).
- The moved coordinate must go toward `x`; the wrong direction yields a farther lattice point.
- This problem uses the **full E8**, not E8P's norm truncation (`||·||² <= 10`) or its 29 hand-picked padding codewords.
