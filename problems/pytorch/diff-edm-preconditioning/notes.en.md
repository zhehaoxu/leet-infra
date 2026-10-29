# Deep Dive: The Four VE/EDM Preconditioning Coefficients and the Churned Euler Step

## Where it comes from

HuggingFace diffusers' `src/diffusers/schedulers/scheduling_edm_euler.py`: `precondition_inputs` (`c_in`), `precondition_noise` (`c_noise = 0.25·ln σ`), `precondition_outputs` (`c_skip` / `c_out` and the recombination), and `step` (churn plus the Euler update). The formulas themselves come from Karras et al. 2206.00364 and NVlabs/edm's `EDMPrecond`. The line `self.timesteps = self.precondition_noise(sigmas)` in `set_timesteps` shows that the EDM family defines its "time conditioning" directly as \(0.25\ln\sigma\), unlike the VP-side \(\bar\alpha\) convention.

## Key ideas and pitfalls

1. **The four coefficients belong to two different sigmas**: `c_in` / `c_noise` use the scheduler's current \(\sigma\), whereas `c_skip` / `c_out` use the post-churn \(\hat\sigma\). Upstream splits it this way because the pipeline calls `scale_model_input` before `step` (no noise added yet), while `precondition_outputs` runs inside `step` with \(\hat\sigma\). Writing \(\hat\sigma\) on both sides is the most insidious bug here: without churn the two coincide and every test passes, but the moment churn is enabled everything is wrong.
2. **The sign of `c_out` encodes the prediction type**: the \(\epsilon\) branch uses \(+\hat\sigma\sigma_d/\sqrt{\hat\sigma^2+\sigma_d^2}\) and the `v_prediction` branch uses the negative. In the EDM paper \(F\) predicts a negatively scaled velocity, so switching to v-prediction only flips the sign; the tests verify it through the identity \(D_v = 2c_{\text{skip}}x - D_\epsilon\).
3. **\(\gamma\)'s denominator is `len(sigmas) - 1`**: `sigmas` already carries the appended terminal 0, so the number of steps is \(K = \text{len}-1\). Using `len(sigmas)` makes the `s_churn` that saturates the cap slightly too small.
4. **The window test uses the pre-churn \(\sigma\)**: `s_tmin <= sigma <= s_tmax`, never \(\hat\sigma\). Outside the window \(\gamma=0\) and `noise` plays no part at all — an implementation that adds noise unconditionally and then multiplies by 0 runs into the 0·NaN trap when the noise contains NaN.
5. **`dt` uses \(\hat\sigma\)**: `dt = sigmas[i+1] - sigma_hat`. Using \(\sigma\) instead inflates the step size and destroys the "lands exactly on `denoised`" property of the last step.
6. **The last-step identity**: with `sigmas[-1] == 0`, \(dt = -\hat\sigma\), so `prev = x + (x - D)/\hat\sigma·(-\hat\sigma) = D`. This is the most useful self-check in the problem: any step whose \(\hat\sigma\) equals `sigmas[i+1]` degenerates into jumping straight to the denoised sample.
7. **The \(\sigma\to t\) map is a monotone logarithmic scale**: `c_noise = 0.25 ln σ`, inverted by \(\sigma = e^{4t}\). That is a different convention from the VP-side \((\bar\alpha,\sigma)\) conversion in `diff-v-prediction-convert`; the two do not overlap.
8. **Numerical precision**: upstream promotes `sample` to float32 (to dodge float16 error on MPS), while this problem computes in float64 and casts back to float32, so the example values match an exact float64 hand calculation.
9. **What is left out**: `preconditioning_threshold` belongs to the training-side `KarrasDenoiser` (SD3's thresholded x₀→v conversion) and does not exist in this scheduler file; to keep the "only algorithms that live in the cited source" discipline, the problem does not invent that branch.

## How the judge cross-checks

The judge uses a pure-Python scalar oracle (`_oracle`) over nested lists that redoes the churn, the four coefficients and the Euler step element-wise, then compares against the reference:

- **Example values**: five output groups across three steps plus churn (`model_input` / `c_noise` / `denoised` / `prev` / `sigma_hat`) are hard-coded to six decimals, and the last step is asserted to have `prev` bitwise equal to `denoised`;
- **Input side decoupled from churn**: with `s_churn=50`, `model_input` still equals "original sample × c_in" rather than "churned sample × c_in"; with \(\gamma=0\), passing `noise` and omitting it give bitwise identical results;
- **Window and clamping**: `s_tmin=5`, `s_tmax=3` and `s_churn=0` all leave \(\hat\sigma=\sigma\); inside the window `s_churn=0.6` gives \(\hat\sigma = 4·(1+0.2)\) and `s_churn=9` is clamped to \(4\sqrt2\);
- **Last step**: `prev_sample == denoised` for three sigma schedules of different lengths;
- **Prediction types**: \(D_v = 2c_{\text{skip}}x - D_\epsilon\) holds at every step;
- **Time scale**: `c_noise` matches \(0.25\ln\sigma\) and `exp(4·c_noise)` recovers \(\sigma\);
- **Combined cross-check**: 3 sigma schedules × every step × 5 configurations (including `v_prediction`, `s_noise=0.7`, `sigma_data=0.2` and in-window churn) agree with the oracle element-wise;
- **Invalid inputs**: 25 groups (non-tensor / integer / mismatched shapes / empty tensors, sigmas as a list / 2-D / integer / length 1 / negative / increasing / NaN, `step_index` out of range or non-integer, `sigma_data=0` or negative, negative `s_churn`/`s_noise`, `s_tmin>s_tmax`, unknown prediction type, churn without noise, noise with the wrong shape or dtype, current \(\sigma=0\)) must all raise `ValueError`.

Eight assertion groups in total.

## Complexity

A constant number of multiply-adds per element of the sample tensor: \(O(B\cdot H\cdot W)\) time and \(O(B\cdot H\cdot W)\) space (one extra copy each for the model input, the denoised output and the next sample).
