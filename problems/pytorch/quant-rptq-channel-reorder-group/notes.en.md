# Quantization: RPTQ Channel Reorder and Group Quantization — Algorithm Notes

## 1. The blind spot of group quantization

The error of group quantization depends on the dynamic range inside each group. If channels are cut in their natural order, adjacent channels may differ by orders of magnitude: one outlier stretches the group scale and the remaining channels keep very few integer grid points, concentrating reconstruction error.

The observation behind RPTQ (Yuan et al., ICLR 2023, arXiv:2304.01089) is that a channel's quantization sensitivity is strongly correlated with its activation range. Putting channels with similar ranges into the same group shrinks the group range substantially. The upstream `quantize/reorder_utils.py` uses sklearn `KMeans(n_clusters, n_init=10, random_state=0)` over 2-D points `(xmax_i, xmin_i)` by default and sorts by cluster label; the same file ships a PEG variant `peg_tensor_calc_reorder_index` that sorts by `xmax-xmin` and splits into equal segments.

## 2. Group quantization formulas

Each group maps `[gmin, gmax]` onto `2^bits` uniform grid points:

\[
s=\operatorname{clamp}\!\left((g_{\max}-g_{\min})\,2^{-b},\ 10^{-4},\ 10^{4}\right)
\]

To align the group midpoint with zero, the zero-point is

\[
z=\operatorname{clamp}\!\left(\operatorname{round}\!\left((g_{\max}+g_{\min})\cdot\frac{-0.5}{s}\right),\ -10^{4},\ 10^{4}\right)
\]

Fake-quant first rounds `w/s`, adds `z`, then clamps to the signed integer interval:

\[
q=\operatorname{clamp}\!\left(\operatorname{round}(w/s)+z,\ -2^{b-1},\ 2^{b-1}-1\right)-z,
\qquad \hat w=q\cdot s
\]

The positive side has only \(2^{b-1}-1\) codes, one fewer than the negative side, so the group's positive maximum saturates first. In the example, `8.4` sits in the group with `scale=1.05`: `8.4/1.05 = 8`, clamped to `7`, reconstructing as `7.35`. `CLIPMIN=1e-4` prevents division by zero for zero-span groups and `1e4` caps scale blow-up.

## 3. Why the PEG variant instead of KMeans

The default upstream path relies on sklearn KMeans floating-point accumulation. Two channels at almost equal distance from a centroid can be assigned to different clusters under torch versus a pure-Python implementation, changing `perm` — fatal for deterministic judging. PEG is the deterministic alternative in the same source file: compare only spans, stable sort, split into equal segments; every language agrees.

This problem uses PEG and fixes the upstream behavior of `counts=[C//g]*g` (which drops the remainder) into balanced segments: the first `C % g` segments get one extra channel, so `sum(counts) == C` and no channel is lost. This is the only intentional deviation from upstream and is stated in the statement.

## 4. Boundaries and degeneracies

- All spans zero: `scale` falls back to `1e-4`. If channel values are also zero, `zero=0` and reconstruction is 0. If the constant is nonzero (e.g. `xmax=xmin=2`), `zero` is clamped to `±1e4` by the formula and reconstruction is determined by `(q-zero)*scale`, which may deviate appreciably from the constant — this is inherent to the upstream formula and is preserved.
- `num_groups=1`: all channels in one group; `scale` follows global max/min.
- `num_groups=C`: one channel per group.
- Input requirements: `bits ∈ [2,8]`, `num_groups ∈ [1,C]`, `xmin <= xmax`, all finite.
- Output dtypes: `perm/counts/zero_points` int64, `scales/dequant` float32; all internal math in float64.

## 5. Division of labor within the collection

- `gptq-column-quantize`: cross-column error compensation (Hessian inverse); this problem does none.
- `quant-squeezellm-sensitive-kmeans`: learns non-uniform grid positions; this problem keeps a uniform grid.
- `quant-double-quant-scale` / `quant-nf4-quantile-grid`: double-quantized scales / quantile grids; this problem uses a fixed exponential grid.
- `llm-int8-outlier-split`: splits channels into outlier and normal parts; this problem **reorders and groups** channels without splitting.
- This problem: deterministic channel reorder (permutation) + balanced grouping + per-group asymmetric fake-quant.

## 6. Validation

- `check_example`: hand-computed example covering positive-side saturation (`8.4 -> 7.35`).
- `check_perm_is_span_sorted`: `perm` is a permutation and is stably ascending in `(span, channel index)`.
- `check_balanced_groups`: segment lengths are only `C//g` or `C//g+1`, with the remainder in the first `C%g` segments.
- `check_group_scales`: each group's `scale/zero` is uniquely determined by that segment's `max/min`.
- `check_matches_oracle`: 40 random configurations compared against a pure-Python scalar oracle (`perm/counts/scale/zero/reconstruction`).
- `check_quantization_grid`: reconstructed values lie on the integer grid and the recovered integer codes are clamped to `[qmin, qmax]`.
- `check_non_identity_and_single_group` / `check_constant_channels`: non-identity reorder, single-group degeneracy, zero spans and constant channels.
- `check_validation`: invalid types, ranges, shapes, finiteness and ordering raise `ValueError`.
