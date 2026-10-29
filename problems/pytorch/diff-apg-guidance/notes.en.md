# In Depth: Adaptive Projected Guidance

## Origin

*Adaptive Projected Guidance* (arXiv:2410.02416). The implementation follows HuggingFace diffusers' `src/diffusers/guiders/adaptive_projected_guidance.py`: `normalized_guidance` (L211–253) fixes the order "momentum → clamp → projection → η blend → extrapolate", and `MomentumBuffer` (L163–170) defines the recursion `running = update + momentum · running`. The same file also ships `adaptive_projected_guidance_mix` and the `use_original_formulation` branch of `AdaptiveProjectedGuidance`; this problem takes the diffusers-native convention (the unconditional prediction is the extrapolation base).

## Core points and pitfalls

1. **The parallel direction comes from the conditional prediction, not from the difference**: \(\hat e=\epsilon_c/\lVert\epsilon_c\rVert\). Normalizing \(\delta\) instead yields a different operator altogether.
2. **The order is not interchangeable**: update the buffer first (it stores the pre-clamp \(\delta\)), then clamp, then project. Clamping before storing poisons the momentum state for the next step, and the judge checks exactly this.
3. **The extrapolation base is the unconditional prediction**: `out = ε_u + s·update`. Using `ε_c + s·update` is the classic mistake and fails the \(\eta=1\) identity assertion outright.
4. **\(\eta\) only scales the parallel part**: the orthogonal coefficient stays at 1, so the update is linear in \(\eta\) and \(\eta=0.5\) is the midpoint of both ends; at \(\eta=1\) we have `δ⊥+δ∥=δ` and the whole thing collapses to bare CFG.
5. **The clamp is a per-sample scalar**: the factor `min(1, τ/‖δ‖)` reduces over every dim but the batch. For a zero difference the norm is 0, so `clamp(max=1)` keeps the factor at 1 — writing `τ/norm` directly gives inf, and inf × 0 becomes NaN.
6. **The normalization eps is 1e-12**: the `F.normalize` default. With a zero conditional prediction the direction is the zero vector and the entire difference counts as the orthogonal component.

## Verification approach

The reference solution vectorizes in float64 (`torch.linalg.vector_norm` plus broadcast inner products) and casts to float32. The judge's `_oracle` first permutes `dims` to the trailing axes, flattens each slice, and recomputes norms, unit vectors and projections with python scalar loops (no tensor reductions), so custom `dims` are covered. Four structural groups: \(\eta=1\) equals bare CFG, the \(\eta=0\) orthogonality plus the linear \(\eta\) interpolation, the post-clamp norm bound with bit-identical results for samples under the threshold, and the momentum recursion including "the clamp never leaks into the buffer". Tolerance `allclose(atol=1e-5)`.

## Complexity

Element-wise \(O(N)\) plus one per-sample norm reduction (also \(O(N)\)), and one extra buffer of the same shape.
