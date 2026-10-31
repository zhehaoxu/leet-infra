# Quantization: E8 Lattice Nearest-Point Rounding — Algorithm Notes

## 1. The blind spot of scalar quantization

Scalar quantization rounds each coordinate independently and accumulates error per coordinate; at a fixed average bit width, higher-dimensional lattices cover the same volume with fewer codewords and thus lower rate-distortion. E8 is one of the densest lattices in 8 dimensions, with the most points per unit volume, making it a natural skeleton for vector-quantization codebooks such as QuIP#'s E8P.

QuIP# (arXiv:2402.04396) constructs E8P as: the 227 codewords of `D8+1/2` with norm at most `sqrt(10)`, plus 29 norm-12 codewords, expanded to `2^16` points by 8 sign flips and `±1/4` shifts. This problem keeps only the central data-independent step: **nearest-point rounding on the full E8**. The 227+29 truncation and the hand-picked codewords depend on fixed data and are not the operator's focus.

## 2. D8 rounding and parity correction

The D8 constraint is "all coordinates integer and the coordinate sum even". First round coordinate-wise to the nearest integer `y = round(x)`:

- If `sum(y)` is even, `y ∈ D8` and we are done;
- If it is odd, exactly one coordinate must move by 1 (moving two coordinates adds error twice and is never better).

Let `d_i = x_i - y_i ∈ [-0.5, 0.5]`. Moving `y_i` by 1 toward x changes the error by

\[
\Delta_i = (1-|d_i|)^2 - d_i^2 = 1 - 2|d_i|
\]

So choose the coordinate with the largest `|d_i|` — it is closest to the midpoint and flipping it to the other integer barely increases error. If several tie for the maximum, this problem takes the smallest index; if `d_i = 0` (x is exactly integer), both directions cost 1 and the convention is `+1`.

## 3. The D8+1/2 coset and the final choice

The other coset of E8 is `D8 + 1/2`. Shift `z = x - 0.5` onto the integer grid, apply the same D8 rounding to z to get `y'`, and the candidate is `y' + 0.5`. The parity test uses the coordinate sum of `y'`; the `+0.5` is only a translation and does not participate.

Finally compare the squared errors `||x - y||²` of the two candidates; choose coset 1 only if strictly smaller, ties go to 0. This is the nearest point in full E8: E8 is the union of the two cosets, so the better of their individual optima is globally optimal.

Upstream `E8P12_codebook.quantize` tries both `X+1/4` and `X-1/4` shifted codebooks. For the **full E8** these shifts are the same lattice: `(1/2,...,1/2)` belongs to D8 (coordinate sum 8 is even), so the difference `±1/2` between the two shifts is itself in E8. Only when the codebook is truncated to E8P's finite set do the two shifts differ; this problem does not truncate and therefore does not compare `±1/4` paths.

## 4. Boundaries and implementation contract

- Coordinate rounding is round-half-to-even (`torch.round`), removing directional ambiguity at x.5.
- Ties for the largest deviation pick the smallest index; a zero deviation uses direction `+1`. These rules only matter for exactly symmetric inputs but must be fixed for deterministic judging.
- Integer points with odd coordinate sum: all `|d_i|=0`, any single move costs 1; the convention yields the point with the smallest index moved by `+1`.
- When the two cosets are equidistant (e.g. every coordinate equals `0.25`), choose coset 0.
- The input must be a finite `[N, 8]` floating tensor; float64 internally, float32 output.

## 5. Division of labor within the collection

- `quant-squeezellm-sensitive-kmeans`: 1-D clustering of sensitive channels with free codeword positions and no lattice constraint;
- `quant-nf4-quantile-grid`: 1-D quantile grids where coordinates are independent;
- `quant-double-quant-scale`: quantizes the scale again, still a 1-D problem;
- This problem: **8-D E8 lattice + coordinate-sum parity constraint**, the only rounding problem in the collection where coordinates are coupled through a parity check;
- Relation to QuIP#: this implements the nearest-lattice rounding underlying the E8P codebook; E8P's truncated codebook and hand-picked padding are out of scope.

## 6. Validation

- `check_example`: hand-computed case covering the parity fix, move direction and coset choice.
- `check_lattice_membership`: coset-0 rows must be integer points with even sum, coset-1 rows half-integer points with even sum; `sse` matches `||x-rounded||²`.
- `check_matches_oracle`: 40 random points at different scales compared against a pure-Python local-enumeration oracle (all of `y` and `y±e_i`).
- `check_parity_fix`: `[1.2, 0, ...]` must correct to `[2, 0, ...]`, validating direction and error.
- `check_coset_selection`: `0.4` is closer in `D8+1/2`, so coset 1 must be chosen.
- `check_tie_prefers_d8`: `0.25` is equidistant from both cosets, so coset 0 must be chosen.
- `check_validation`: wrong dimensionality, shape, floating type or non-finite values raise `ValueError`.
