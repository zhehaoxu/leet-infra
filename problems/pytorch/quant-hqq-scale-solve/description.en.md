# Quantization: HQQ Half-Quadratic Closed-Form Scale Solver

## Problem

RTN (round-to-nearest) fixes the scale from `absmax` first and then rounds the weights with it — the scale is chosen **a priori**, and the rounding error never enters any objective. **HQQ** (Badri & Shaji, 2024, arXiv:2403.07134) rewrites quantization as an **alternating optimization**:

\[
\min_{s,\ z,\ q}\ \|W-s\,(q-z)\|_2^2
\quad\text{s.t.}\quad q=\operatorname{round}(W/s)+z,\ \ z\in\{0,\dots,2^b-1\}
\]

With \(q\) and \(z\) fixed (both integer), the optimal \(s\) is **closed-form** (least squares / Wiener solution):

\[
s^\star=\frac{\langle W,\ q-z\rangle}{\|q-z\|_2^2}
\]

So each group needs only a few "round → closed-form scale" iterations to reach a reconstruction error below RTN. The zero-point \(z\) is aligned to the group minimum (\(z=\operatorname{clamp}(\operatorname{round}(-W_{\min}/s),0,2^b-1)\)); this problem does not search over \(z\).

## Task

Implement `solve(W, bits, iters)`:

- Input: `W` float32 `(G, N)` (one group per row), `bits` an integer in 2..8, `iters` a non-negative integer (number of alternating iterations).
- Per group, independently:
  1. `lo, hi = W.min(), W.max()`; if `hi == lo` (constant group): `scale = 1.0`, `zero = 0`, reconstruction is `lo` everywhere, skip the iterations;
  2. Initialize `s = (hi - lo) / (2^bits - 1)`, `z = clamp(round(-lo / s), 0, 2^bits-1)`;
  3. Repeat `iters` times: `q = clamp(round(W/s) + z, 0, 2^bits-1) - z`, then the closed-form `s = <W,q> / <q,q>` (keep `s` if `<q,q> == 0`);
  4. Recompute `q` once with the final `s`, then reconstruct `deq = q * s`.
- Return `(scale, zero, deq)`: `scale` float32 `(G,)`; `zero` int64 `(G,)`; `deq` float32 `(G, N)`.

**Conventions**: float64 internally, float32 output; tolerance \(10^{-5}\). Out-of-range `bits` or `iters < 0` raises `ValueError`.

## Example

```
W = [[0, 1, 2, ..., 15]], bits = 4, iters = 2
lo=0, hi=15 → s=1, z=0; q = W exactly (integer lattice); s = <W,W>/<W,W> = 1
Output: scale=[1.0], zero=[0], deq=[[0, 1, 2, ..., 15]]
```

## Constraints

- The closed-form scale must be \(\langle W,q\rangle/\langle q,q\rangle\), not a fallback to `(hi-lo)/(2^b-1)`.
- Once initialized from the min, `z` stays fixed (HQQ's simplified variant: no zero-point search).
- Constant groups follow the degeneracy rule above (`scale=1, zero=0, deq=lo`).
