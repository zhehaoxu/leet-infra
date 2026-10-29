# Deep Dive: LCM Boundary-Condition Scalings and the Few-Step Denoise Step

## Source

HuggingFace diffusers, `src/diffusers/schedulers/scheduling_lcm.py`: `get_scalings_for_boundary_condition_discrete` (\(c_{\text{skip}}\) / \(c_{\text{out}}\) with \(\sigma_d\) hard-coded to 0.5), `step` (\(x_0\) conversion, clip/threshold, boundary-condition blend, re-noising with \(\bar\alpha_{\text{prev}}\)), and `set_timesteps` (the LCM distillation grid `arange(1, original_steps+1)*k - 1`, subsampled via `floor(linspace(0, N, endpoint=False))`). The algorithm is from Luo et al., 2310.04378 (Latent Consistency Models) and the `get_scalings_for_boundary_condition_discrete` of its official implementation.

## Key Points and Pitfalls

1. **\(c_{\text{out}}\) is missing \(\sigma_d\)**: LCM uses \(s/\sqrt{s^2+\sigma_d^2}\), EDM uses \(s\sigma_d/\sqrt{s^2+\sigma_d^2}\). With \(\sigma_d=0.5\) the two differ by exactly a factor of 2, and the distilled LCM weights were trained against this coefficient — **do not "fix" it back to the EDM formula**, or every output is off by 2×. Corollary: the EDM identity \(c_{\text{skip}}^2+c_{\text{out}}^2=c_{\text{skip}}\) does not hold for LCM.
2. **The coefficients come from the timestep index, not from \(\sigma\)**: \(s=t\cdot\text{timestep\_scaling}\), with `timestep_scaling` defaulting to 10. LCM timesteps are indices into the training grid (0..999), so \(s\) spans 0..9990; treating them as continuous time \(t\in[0,1]\) gives entirely different coefficients.
3. **\(\sigma_d\) is assigned inside the function body**: upstream writes `self.sigma_data = 0.5` inside `get_scalings_for_boundary_condition_discrete`, overwriting it on every call and making it non-configurable. This problem copies that constant.
4. **The \(t=0\) degeneracy**: \(c_{\text{skip}}=1\), \(c_{\text{out}}=0\), so `denoised` equals the input sample exactly, independent of the model output. The real LCM grid starts at \(k-1\) (\(k=T//\text{original\_steps}\), e.g. 19 for 1000/50), never 0, so actual inference avoids it; this problem permits 0 in the schedule and pins it down with an explicit assertion.
5. **`prev_timestep` is the next array entry, not \(t-1\)**: `timesteps[step_index+1]`. LCM's grid is an evenly spaced subsample (spacing \(k\)), so \(t-1\) would land on a different noise level. On the final step there is no next entry and the current timestep is reused — a value that takes no part, since `denoised` is returned directly.
6. **The final step does not re-noise**: the test is `step_index != num_inference_steps - 1`, using the **configured** step count rather than "is the current timestep 0". Hence one-step sampling (a length-1 schedule) returns pure `denoised` — precisely what LCM 1-step inference is.
7. **The \(\bar\alpha_{\text{prev}}\) lookup guards against negative indices**: `alphas_cumprod[prev_timestep] if prev_timestep >= 0 else final_alpha_cumprod`. This problem sets `final_alpha_cumprod` to 1.0 (upstream's `set_alpha_to_one=True` default), so a negative timestep yields \(\sqrt{1}=1\) and \(\sqrt{0}=0\), i.e. simply returns `denoised`.
8. **The \(x_0\) conversion uses the correct \(\alpha_t\) form**: the `v_prediction` branch is \(\sqrt{\bar\alpha_t}x-\sqrt{1-\bar\alpha_t}v\), unlike the missing-\(\alpha\) version in `diff-euler-ancestral-step`. Two v-prediction conversions in the same repository disagree — worth reading side by side.
9. **Clipping has two layers**: `thresholding=True` applies dynamic thresholding, otherwise `clip_sample=True` applies fixed-range clipping (default \([-1,1]\)). Dynamic thresholding already has its own problem (`diff-dpm-x0-conversion-threshold`), so this one keeps only `clip_sample`, applied to \(x_0\) and **before** the blend.
10. **Numerics**: upstream computes and returns in `model_output.dtype`; this problem computes in float64 internally and returns float32.

## Reference Comparison

Judging uses a numpy `_oracle` that redoes the coefficients, \(x_0\), clipping, blending, and re-noising (with inputs rounded to float32, the same rounding the reference solution sees):

- **Example values**: `c_skip`, `c_out`, `denoised`, and `prev` at `t=9` / `t=2` are hard-coded to six decimals, and the final step asserts `prev` is bitwise equal to `denoised`;
- **Coefficient table**: recompute \(c_{\text{skip}}\) / \(c_{\text{out}}\) in closed form, assert \(c_{\text{out}}=2\times\) the EDM value, assert the EDM identity \(c_{\text{skip}}^2+c_{\text{out}}^2=c_{\text{skip}}\) fails, check monotonicity over four timesteps (\(c_{\text{skip}}\) down, \(c_{\text{out}}\) up), the degenerate values at \(t=0\) together with `denoised==sample`, and the relation between `timestep_scaling` 5 and 10;
- **Prediction types**: back out \(\epsilon\) and \(v\) from one \(x_0\) and confirm all three branches give the same `denoised`; the `sample` branch equals \(c_{\text{out}}\cdot\text{model}+c_{\text{skip}}\cdot x\);
- **Clipping**: on versus off, a tighter `clip_sample_range=0.4`, and clipping happening before the blend;
- **Re-noising**: with zero noise `prev=√ᾱ_prev·denoised`, with unit noise the difference is exactly \(\sqrt{1-\bar\alpha_{\text{prev}}}\), the final step is invariant to any noise, and a one-step schedule is the final step;
- **Where `prev_timestep` comes from**: the same current timestep with schedules of different lengths gives different `prev` but identical `denoised`; repeated timesteps take the next array entry rather than \(t-1\);
- **Combined comparison**: 2 shapes × 4 schedules (one-step, repeated values, containing 0) × every step × 3 prediction types, plus a round with `clip_sample=False` and one with `clip_sample_range=0.3`;
- **Invalid inputs**: 21 cases (non-tensor / integer / mismatched shapes / `alphas_cumprod` as a list / 2-D / containing 1.0 / containing 0 / `timesteps` as a list / float / out of range / negative / 2-D, out-of-range or `bool` `step_index`, unknown prediction type, `clip_sample=1`, `clip_sample_range=0`, negative `timestep_scaling`, missing `noise` on a non-final step, wrong `noise` shape) must all raise `ValueError`.

Eight assertion groups in total.

## Complexity

A constant number of multiply-adds per element: \(O(B\cdot C\cdot H\cdot W)\) time and space (one copy each for `denoised` and `prev_sample`); the coefficients are scalars independent of tensor size.
