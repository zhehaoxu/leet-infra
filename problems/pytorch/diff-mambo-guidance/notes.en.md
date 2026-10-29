# Deep Dive: MAMBO-G Magnitude-Aware Guidance

## Where it comes from

*Magnitude-Aware Mitigation for Boosted Guidance* (arXiv:2508.03442). The implementation follows HuggingFace diffusers' `src/diffusers/guiders/magnitude_aware_guidance.py`: `mambo_guidance` (L141–159) is the whole thing — it takes `dim = list(range(1, len(pred_cond.shape)))`, computes `torch.norm(diff, dim=dim, keepdim=True) / torch.norm(pred_uncond, dim=dim, keepdim=True)`, then `guidance_scale_final = 1.0 + (guidance_scale - 1.0) * torch.exp(-alpha * ratio)`. The same function's `use_original_formulation` branch swaps the base to `pred_cond` and the coefficient to `guidance_scale * exp(-alpha * ratio)`; this problem sticks to the diffusers-native formulation (base \(\epsilon_u\)).

## Key ideas and pitfalls

1. **The norm is a per-sample scalar, not per element**: the reduction runs over `range(1, ndim)`. An elementwise ratio \(|\delta_i|/|u_i|\) would give every element its own \(w'\), which is a different operator.
2. **The decay acts on the effective scale, not on the result**: the correct form is `out = u + w'·δ`, where \(w'\) already carries the offset of 1. Writing `out = w'·(u + w·δ)` rescales the unconditional prediction too.
3. **The constant 1 in \(w'=1+(w-1)e^{-\lambda r}\) is not optional**: dropping it gives \(w'=w\,e^{-\lambda r}\), and then \(w=1\) no longer collapses to the identity, failing the `scale=1` assertion immediately.
4. **Handle the zero-norm case as \(r\to+\infty\)**: with \(\lambda r\to+\infty\) the decay \(e^{-\lambda r}\to 0\) gives \(w'=1\) for free. Computing \(0/0\) yields `NaN`, and computing `inf` before multiplying by `(w-1)` yields `inf`/`NaN`; either poisons the whole sampling trajectory.
5. **\(\lambda=0\) must reduce exactly to CFG**: \(e^{0}=1\), so \(w'=1+(w-1)=w\). This is the boundary that is easy to get right and just as easy to break (for example by writing the coefficient as \(w\,e^{-\lambda r}\)).
6. **The direction never changes**: MAMBO-G only rescales, so \((out-u)/(c-u)\) is a constant within each sample — the judge checks this property explicitly.

## How the judge cross-checks

The reference solution computes both per-sample norms in float64 with `torch.linalg.vector_norm`, zeroes the decay factor of zero-norm samples via a boolean mask, and casts to float32. The `_oracle` flattens each sample into a python list and recomputes both norms plus `math.exp` with scalar loops (`w'=1` directly when `norm_uncond == 0`), reusing no tensor reductions. Five structural assertion groups: \(\lambda=0\) equals plain CFG, \(w=1\) is a bit-exact identity, \((out-u)/(c-u)\) is constant per sample and matches the analytic \(w'\), \(w'\) decreases monotonically in \(\lambda\) and converges to 1, and zero-norm samples stay free of `NaN`. Tolerance `allclose(atol=1e-5)`.

## Complexity

Two elementwise passes (for the two norms) plus one write pass, \(O(N)\) with no extra buffers; the norms reduce over every dimension except the batch.
