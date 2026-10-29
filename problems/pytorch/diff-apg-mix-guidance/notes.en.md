# Deep Dive: APG-Mix CFG Warmup and Orthogonal Projected Guidance

## Where it comes from

*Adaptive Projected Guidance* (arXiv:2410.02416). The implementation follows HuggingFace diffusers' `src/diffusers/guiders/adaptive_projected_guidance_mix.py` (`forward`, `normalized_guidance`, `MomentumBuffer`), the guider used by HunyuanImage-2.1's sampling pipeline. Plain APG lives in `guiders/adaptive_projected_guidance.py`; the two share the projection operator and differ only in whether momentum is warmed up by CFG before switching to projection. APG itself is already covered by `diff-apg-guidance` (order 501) — this problem only tests the **two-phase assembly plus the cross-step momentum bookkeeping**.

Upstream has one naming quirk we drop: the guider config field `adaptive_projected_guidance_rescale` (default 10.0) performs no rescaling at all; it is passed positionally as `norm_threshold` into `normalized_guidance`. We simply call it `norm_threshold`.

## Key ideas and pitfalls

1. **The CFG branch uses the unsmoothed difference**: the source calls `update_momentum_buffer(...)` first, then still shifts by `shift = pred_cond - pred_uncond`. Swapping `shift` for `running` is the most common mistake: the warmup phase would then depend on history, although APG is not active yet and the output must be bit-identical to plain CFG.
2. **The buffer stores the pre-clamp value**: `normalized_guidance` updates the buffer first and only then copies `diff` out for norm clamping. Writing the clamped value back into the buffer compresses the momentum every step, and the error accumulates (the 10-step chained cross-check is designed to catch this).
3. **APG starts strictly after the threshold**: `step > adaptive_projected_guidance_start_step`. Using `>=` switches to projection one step early at `apg_start_step=5`.
4. **APG is gated on CFG**: `_is_apg_enabled` begins with `if not self._is_cfg_enabled(): return False`. So when `cfg_scale` degenerates (`isclose(cfg, 1.0)` by default), the result is `pred_cond` even if `apg_scale` is fine — and the momentum is **not** updated.
5. **Norms run over every dimension except the batch**: the source builds `dim = [-i for i in range(1, len(diff.shape))]`, i.e. `(C,H,W)` for latents of shape `(B,C,H,W)` — one norm per sample, not per pixel. Using `dim=-1` clamps each pixel independently and completely changes the projected direction.
6. **Negative momentum**: the default is `-0.5`, so `running = d + \mu\cdot running\_prev` **extrapolates** between the current and previous difference (`0.5d`). Treating it as an EMA, `(1-\mu)d + \mu\,prev`, gives a different number.
7. **No momentum update outside the window**: the `not cfg_on` branch returns `pred_cond` without touching the buffer; if the incoming `buffer` is `None`, return `None` (no state was ever established) rather than fabricating a zero tensor.
8. **`use_original_formulation` changes two things**: the base moves from `pred_uncond` to `pred_cond`, and the degeneracy test flips from `isclose(s, 1.0)` to `isclose(s, 0.0)`. Changing only the base leaves guidance running at `cfg_scale=0.0`.

## How the judge cross-checks

The reference solution computes everything in float64 and returns float32, carrying the momentum state as float32 (as the real pipeline does, which introduces ~\(10^{-7}\) rounding). The test `_oracle` reuses no tensor ops: windows are re-derived with python `int()` and `math.isclose`, while momentum recurrence, norms, inner products and the orthogonal split are all per-element python scalar loops (reshape to `(batch, width)` first). Eight assertion groups: the three given example rows; the warmup phase ("steps 0–5 are bit-identical to CFG while momentum recurs per step"); three momentum configurations (`None` / `0.0` / negative) plus no in-place mutation of the input buffer; threshold clamping (off, firing, scaling-only); projection properties (\(\eta=1\) degenerates to CFG, \(\eta=0\) is orthogonal to \(\epsilon_c\), norm never grows); window and `use_original_formulation` semantics; four parameter sets chained over 10 steps (including a partial window); and invalid inputs. Tolerance `atol=1e-5`.

## Complexity

Elementwise \(O(\text{numel})\) per step plus two per-sample reductions (norm, inner product); the momentum state costs one extra prediction-shaped buffer that lives across steps.
