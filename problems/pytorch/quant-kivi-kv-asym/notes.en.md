# Quantization: KIVI Asymmetric KV Cache Quantization — Algorithm Notes

## 1. Why keys and values need different granularities

The core observation of KIVI (Liu et al., ICML 2024, arXiv:2402.02750) is that keys and values in the KV cache differ in the **directionality** of their distributions:

- A few key feature channels (a handful of dims along `head_dim`) carry most of the variance, and these "outlier channels" stay at **the same positions across all tokens**. Per-token quantization lets every token's own outlier eat code space, while per-channel quantization only needs high resolution for those few channels.
- Values have no fixed outlier channels, but their overall scale varies strongly across tokens (especially early ones). Per-token quantization lets each token set its own scale.

Hence KIVI's recipe: "keys per-channel + values per-token", matching this problem's signature.

## 2. The asymmetric convention (judged point)

This problem uses KIVI's simplest zero-point form — **the zero-point is the group's min**:

\[
\Delta=\frac{x_{\max}-x_{\min}}{2^b-1},\qquad
q=\operatorname{clamp}\!\left(\operatorname{round}\!\left(\frac{x-x_{\min}}{\Delta}\right),0,2^b-1\right),\qquad
\hat x=q\Delta+x_{\min}
\]

- \(x_{\min}\) lands exactly on code 0 and \(x_{\max}\) exactly on code \(2^b-1\) — `check_roundtrip_bound` asserts lossless min/max per group; any implementation that divides by \(2^b\) instead of \(2^b-1\), or rounds the zero-point, breaks it.
- If the zero-point were implemented as an integer `round(-x_min/Δ)`, groups far from zero (e.g. \([2,3]\)) would push it outside \([0,2^b-1]\), get clamped, shift the whole covered range and blow up the error. Using the floating-point min avoids this trap; the \([1,3]\) and \([2,3]\) groups in `check_example` exist exactly for this.
- Degenerate groups (`max == min`): `Δ = 1`, all codes 0, `deq = x_min`, lossless — `check_degenerate_and_validation` asserts it with constant channels/tokens.

## 3. How the granularity assertions are written

`check_granularity` pins down two things at once:

1. Parameter tensor **lengths**: K's scale/zero are `(D,)`, V's are `(T,)` — swapping the reduction axis fails on shape immediately;
2. Parameter tensor **values**: `ks == (K.amax(0) - K.amin(0))/15`, `kz == K.amin(0)`, `vs == (V.amax(1)-V.amin(1))/15`, `vz == V.amin(1)` — so even when `T == D` coincidentally, the values still discriminate.

## 4. Division of labor within the collection

- `per-channel-quantize`: **symmetric** (absmax/127) per-channel on a generic weight axis;
- `asymmetric-quantize`: the generic symmetric-vs-asymmetric comparison on a single tensor;
- This problem: a **combination of two granularities** (K per-channel / V per-token) plus a floating-point-min zero-point, grounded in the KV cache setting;
- `kv-cache-memory`: the KV cache footprint formula, no quantization numerics.

## 5. Implementation notes

- Vectorize: `amin/amax(dim, keepdim=True)` then broadcast `(x - mn)/scale`, `round().clamp(0, qmax)`, and `q*scale+mn`; make parameters 1-D with `squeeze(dim)`.
- `torch.round` is round-half-to-even (`round(2.5) == 2`), consistent with the paper's implementation; do not use `floor(x+0.5)`.
- Handle degenerate groups with `torch.where(span > 0, span/qmax, ones)` to avoid NaN from `0/0`.

## 6. Verification

- `check_example`: \(2\times2\) hand-computed scale/zero and lossless round trip.
- `check_granularity`: both length and value assertions with \(T\neq D\) (\(5\times7\)).
- `check_roundtrip_bound`: error ≤ Δ/2, group min/max exactly restored.
- `check_matches_oracle`: \(4\times3\), 6 bits, per-group scalar recomputation.
- `check_degenerate_and_validation`: constant groups restored exactly + out-of-range `bits` raising `ValueError`.
