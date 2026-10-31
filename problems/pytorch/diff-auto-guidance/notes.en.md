# Notes: AutoGuidance Flips Its Anchor and Its Identity Threshold Together

## Source

- HuggingFace diffusers v0.42.1, `src/diffusers/guiders/auto_guidance.py`: the paper link is on line 34 of the class docstring, `_input_predictions` on line 64, `forward` on lines 156-170, `num_conditions` on lines 176-181, `_is_ag_enabled` on lines 182-198.
- Same library, `src/diffusers/guiders/guider_utils.py`: the `start / stop` validation inside `BaseGuidance.__init__` is on lines 59-63, `set_state` on lines 100-104, `rescale_noise_cfg` on lines 374-396.
- Paper: Karras et al., "Guiding a Diffusion Model with a Bad Version of Itself", arXiv:2406.02507 (title and authors verified). The standard-deviation pull-back comes from "Common Diffusion Noise Schedules and Sample Steps are Flawed", arXiv:2305.08891 (also verified).

## Core Points and Pitfalls

- **One switch moves two things.** `use_original_formulation` relocates both the extrapolation anchor (`pred_uncond` -> `pred_cond`) and the identity scale (1.0 -> 0.0). They must move together, as in `isclose(s, 0.0 if original else 1.0)`. Flipping only the anchor turns "guidance off at s=0" into "output carries one extra shift"; flipping only the threshold treats `s=1` as off when it is really one `pred_cond - pred_uncond` away.
- **The two faces of `math.isclose`.** With the defaults `rel_tol=1e-9, abs_tol=0`, comparing against 1.0 is relative: `1+1e-10` counts as identity, `1+1e-8` does not. Comparing against 0.0 makes `rel_tol * |b|` vanish, so only exactly `0.0` is identity and `1e-12` stays enabled. `check_isclose_is_relative` exercises both, and the original-formulation `s=1e-12` case can only be compared with an absolute tolerance (it differs from `pred_cond` at the 5e-13 level).
- **The identity branch still runs the rescale.** Upstream keeps `if guidance_rescale > 0.0` outside the if/else, so a disabled guide feeds `pred_cond` into `rescale_noise_cfg`. Mathematically `std(pred_cond)/std(pred_cond) = 1`, i.e. a no-op; in floating point `r*x + (1-r)*x` is bitwise exact for binary-friendly coefficients such as `r=0.5` but can drift by 1 ulp for `r=0.3`. `check_rescale_runs_on_off_branch` pins both cases, once with `torch.equal` and once with `atol=1e-7`.
- **The window is an int-truncated, left-closed right-open range.** `int(0.1*20)=2` and `int(0.8*20)=16`, so the enabled steps are `[2,16)`. `start` and `stop` are validated at construction time (`start` strictly below 1, `stop` not below `start`), while `step` only participates in the runtime comparison - the two checks live in different functions.
- **Whether the weak branch is computed at all depends on `num_conditions`**: lines 176-181 decide the number of forward passes, and when guidance is off `forward` receives only `pred_cond`, so `pred_uncond` never even gets a shape check. That is exactly why the statement allows `None` - or even a wrongly shaped tensor - on the disabled branch.

## Deviations from Upstream

- Upstream produces the "bad model" prediction by installing `_apply_layer_skip_hook` on a few blocks with a `dropout` probability. Here that degraded prediction is an explicit `pred_uncond` argument, so the hook, `LayerSkipConfig`, and the whole `prepare_models / cleanup_models` assembly disappear.
- The `enabled` flag of `BaseGuidance` is dropped (it is just a third boolean ANDed into `within / close`), and `GuiderOutput` is reduced to the final prediction.
- All intermediate math is forced to float64 and the result is cast to float32; upstream computes in whatever dtype it is given. At least 2 elements outside the batch dimension are also required: a 1-D input or a `(B,1)` input makes `rescale_noise_cfg` hit degrees of freedom <= 0 and return NaN upstream, while this problem raises `ValueError` instead.
- `guidance_rescale` has no upper bound in the source; this problem narrows it to `[0, 1]`, following the sibling `diff-guidance-rescale`. `start / stop` keep the `BaseGuidance.__init__` semantics, where `start` may not be 1.0.

## Division of Labor

- `diff-cfg` drills the single formula `eps_u + w*(eps_c - eps_u)`. Here the whole difficulty is the paired anchor/threshold flip plus the relative-versus-absolute asymmetry of `isclose`; that formula is only the shell.
- `diff-guidance-rescale` owns the standard-deviation pull-back itself (closed interval on a continuous `t`, configurable `dims`, the phi mix). This problem only uses it as an optional tail step, with `dims` fixed to every dimension except the batch one.
- `diff-pag-dual-guidance` and `diff-ltx2-guidance` handle multi-component composites with their own windows; this one is the single-component, two-formulation control case. `diff-mambo-guidance` adapts the scale from a magnitude ratio, whereas the scale here is fixed.
