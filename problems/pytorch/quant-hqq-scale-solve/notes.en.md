# Quantization: HQQ Half-Quadratic Closed-Form Scale Solver — Algorithm Notes

## 1. The blind spot of RTN

RTN derives the scale from the extremes a priori: \(s=(x_{\max}-x_{\min})/(2^b-1)\), then \(q=\operatorname{round}(x/s)\). One outlier inflates the scale and wastes the whole group's resolution, and the rounding error never enters an objective function.

HQQ (Badri & Shaji, 2024, arXiv:2403.07134) turns this into a joint minimization over \((s, z, q)\):

\[
\min_{s,z,q}\ \|W-s(q-z)\|_2^2,\qquad q=\operatorname{round}(W/s)+z
\]

and solves it by **half-quadratic splitting**: fix \(q,z\) and solve \(s\) in closed form; fix \(s,z\) and round for \(q\). This problem implements the first half — the **closed-form scale** — with `z` aligned to the min and then fixed.

## 2. Deriving the closed-form scale (judged point)

For fixed integer \(q\) and \(z\), let \(u=q-z\); the objective is quadratic in \(s\):

\[
\|W-su\|^2=\|W\|^2-2s\langle W,u\rangle+s^2\|u\|^2
\]

Setting the derivative in \(s\) to zero:

\[
s^\star=\frac{\langle W,u\rangle}{\|u\|^2}
\]

This is the classic Wiener filter / least-squares solution. It guarantees that, **for the same integer codes \(u\)**, no other scale is smaller.

`check_closed_form_optimality` asserts it from two angles:

1. Recover the integer codes \(u=\operatorname{round}(\hat W/s)\) from the returned `deq` and verify \(s\approx\langle W,u\rangle/\langle u,u\rangle\) (within \(10^{-4}\) of floating-point round-trip error);
2. Perturb \(s\) by \(\pm10\%\) while keeping \(u\) fixed and recompute the SSE: it must **not decrease** — direct evidence of least-squares optimality. A lazy `(hi-lo)/(2^b-1)` fails the first check.

## 3. Why the zero-point is "min-aligned then fixed"

Full HQQ also grid-searches \(z\) (8 candidates, keep the best). This problem deliberately drops that step for two reasons:

- Determinism: with fixed \(z\) the whole procedure is a deterministic iteration, independent of implementation details (candidate count, tie-breaking);
- Focus: the closed-form scale is the essential difference from RTN. Setting \(z=\operatorname{clamp}(\operatorname{round}(-W_{\min}/s),0,2^b-1)\) places the smallest weight near code 0, so the support of \(u\) covers the data range.

`check_beats_rtn` gives a macro-level guarantee: on the same skewed data, HQQ's total SSE is no larger than symmetric RTN (`absmax/(2^(b-1)-1)`, zero=0). This is not an identity (the two paths produce different \(q\)), but it is a stable empirical fact on skewed normal data and the paper's central claim.

## 4. Degeneracy and boundaries

- Constant group (`hi == lo`): `s=1, z=0, deq=lo`, skip iterations — otherwise `<q,q> = 0` blows up the closed-form division;
- `iters=0`: one initial round plus one reconstruction, i.e. "asymmetric RTN"; allowed for comparison;
- Outside `bits ∈ [2,8]`, `iters ≥ 0`: raise `ValueError`;
- Everything runs in float64: accumulating `<W,q>` in float32 loses precision and destabilizes the \(10^{-4}\) assertion in `check_closed_form_optimality`.

## 5. Division of labor within the collection

- `symmetric-quantize` / `asymmetric-quantize`: fake-quant semantics with a fixed scale;
- `gptq-column-quantize`: **cross-column** error compensation (Hessian inverse);
- `quant-squeezellm-sensitive-kmeans`: optimizes the **grid positions** (non-uniform); here the grid is uniform and only the scale is optimized;
- This problem: **closed-form optimality of the scale** within a group plus the alternating iteration.

## 6. Verification

- `check_example`: exact round trip on the \([0,\dots,15]\) integer lattice.
- `check_closed_form_optimality`: recover \(u\), verify \(s=\langle W,u\rangle/\langle u,u\rangle\), and check perturbations do not help.
- `check_beats_rtn`: \(4\times128\) skewed data, total SSE ≤ symmetric RTN.
- `check_matches_oracle`: 5 bits, 3 groups, pure-python scalar recomputation.
- `check_degenerate_and_validation`: constant-group degeneracy + three kinds of invalid arguments.
