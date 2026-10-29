# Deep Dive: LTX-2 Three-Term Delta Guidance

## Where it comes from

The LTX-2 guider follows HuggingFace diffusers' `src/diffusers/guiders/ltx2_guidance.py`: `forward` spells out the delta form `x0_cond + (gs-1)*(cond-uncond) + stg*(cond-stg) + (mod-1)*(cond-modality)`, each term gated by `is_cfg_enabled` / `is_stg_enabled` / `is_modality_enabled`, all sharing the half-open `_within_window`. STG's paper is *Spatio-Temporal Guidance for Video Diffusion* (arXiv:2411.18664); the four-branch skip-layer skeleton on its own is already covered by `diff-pag-dual-guidance` (order 504), so this problem only asks for LTX-2's distinctive three-term delta form.

## Key ideas and pitfalls

1. **Everything is measured against the conditional prediction**: CFG's native form is `uncond + s*(cond-uncond)`, which algebraically equals the delta form `cond + (s-1)*(cond-uncond)` — but only for a **single** term. Stacking three terms each with its own base (`uncond` / `stg` / `modality`) drifts away from the source; upstream starts from \(\epsilon_c\) throughout, which is what makes the terms additive.
2. **The identity points differ**: `cfg_scale` and `modality_scale` are identity at 1.0, `stg_scale` is identity at 0.0. Those three defaults (1.0 / 0.0 / 1.0) mean "term off", and the source tests them with `math.isclose`, so `1.0 + 1e-12` also counts as off.
3. **Switches are `and not isclose`, not "multiply the difference by zero"**: the difference shows up when \(s_c=1\) still requires the unconditional pass. The judge asserts both directions — `None` is accepted when a term is off, and passing `None` for an active term must raise — which is exactly what kills a "compute then multiply by zero" implementation.
4. **There is only one window**: unlike PAG/SLG, which have a CFG window and a perturbation window, LTX-2's two guiders (video / audio) each hold their own scales while sharing a single strict left-closed, right-open window for all three terms.
5. **`stg_scale` may be negative**: reversed STG guidance is used in practice to damp over-strong temporal consistency, so this problem does not restrict STG to be non-negative, and allows any finite real for the CFG/modality scales too (no lower bound).
6. **Multi-modality instantiation is out of scope**: in the real pipeline video and audio each get a guider instance with their own scales, and `spatio_temporal_guidance_blocks` is read only from the video guider. That is assembly detail; this problem is the single-modality single step.

## How the judge cross-checks

The reference solution computes in float64 and returns float32. The test `_oracle` redoes the window and the three-term accumulation with python scalars (reshape to `(B,-1)`, then elementwise). Seven assertion groups: the three example rows (including the default degenerate case), identity-point degeneracy (all three branch predictions may be omitted), additivity (`full - cond == Σ deltas`, with each single-term run starting from `cond`), scale semantics (a CFG/modality scale of 2.0 adds one difference, an STG scale of 1.0 adds one difference; `isclose` degeneracy), window boundaries (left-closed right-open; no branch predictions needed outside), five parameter sets swept over 10 steps, and invalid inputs. Tolerance `atol=1e-5`.

## Complexity

Elementwise \(O(\text{numel})\); each of the four predictions is read once, with no reductions.
