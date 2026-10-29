# LCM Boundary-Condition Scalings and the Few-Step Denoise Step

## Problem Statement

Latent Consistency Models (LCM) distill a diffusion model into a sampler that needs four steps — or even one. They reuse EDM's **boundary-condition preconditioning**: the network's \(x_0\) prediction is not used directly but blended with the current sample through two coefficients \(c_{\text{skip}}\) and \(c_{\text{out}}\), so that the denoised result automatically satisfies the consistency condition at both ends of the noise range. The coefficients depend only on the **discrete timestep**, stretched by `timestep_scaling` (default 10).

This problem implements one full step of `LCMScheduler.step`: derive the boundary-condition coefficients from the timestep, convert \(x_0\) per prediction type, clip, blend, then re-noise using the next step's \(\bar\alpha\).

## Rules

Let \(\bar\alpha_t = \text{alphas\_cumprod}[t]\) and \(t = \text{timesteps}[i]\) (\(i\) = `step_index`).

**1. Boundary-condition coefficients** (\(\sigma_d\) is hard-coded to 0.5 upstream):

\[ s = t\cdot\text{timestep\_scaling},\qquad c_{\text{skip}} = \frac{\sigma_d^2}{s^2+\sigma_d^2},\qquad c_{\text{out}} = \frac{s}{\sqrt{s^2+\sigma_d^2}} \]

Note \(c_{\text{out}}\) is **not** multiplied by \(\sigma_d\) — that is the difference from EDM.

**2. \(x_0\) conversion** (via the square roots of \(\bar\alpha_t\)):

- `epsilon`: \(x_0 = \dfrac{x-\sqrt{1-\bar\alpha_t}\,\epsilon}{\sqrt{\bar\alpha_t}}\)
- `sample`: \(x_0 = \text{model\_output}\)
- `v_prediction`: \(x_0 = \sqrt{\bar\alpha_t}\,x-\sqrt{1-\bar\alpha_t}\,\text{model\_output}\)

With `clip_sample=True`, \(x_0\) is then clamped to \([-\text{clip\_sample\_range}, \text{clip\_sample\_range}]\).

**3. Blend**: \(\text{denoised} = c_{\text{out}}x_0 + c_{\text{skip}}x\).

**4. Re-noising**: let \(\bar\alpha_{\text{prev}} = \text{alphas\_cumprod}[\text{timesteps}[i+1]]\) (if that entry does not exist, the current \(t\) is reused; that value plays no part on the final step; a negative index yields 1.0).

- Not the final step: \(\text{prev} = \sqrt{\bar\alpha_{\text{prev}}}\,\text{denoised} + \sqrt{1-\bar\alpha_{\text{prev}}}\,\text{noise}\)
- Final step (\(i = \text{len(timesteps)}-1\)): \(\text{prev} = \text{denoised}\), no `noise` consumed

`noise` is an explicit argument (a float tensor shaped like the sample).

Return `(denoised, prev_sample, c_skip, c_out)`: the first two are float32 tensors shaped like `sample`, the last two are python floats.

## Example

With `alphas_cumprod=[0.99, 0.9, 0.7, 0.5, 0.3, 0.15, 0.06, 0.02, 0.005, 0.001]`, `timesteps=[9, 5, 2]`, `sample=[[1.0, -2.0, 0.5], [0.25, 3.0, -1.5]]`, `model_output=[[0.5, 0.25, -0.75], [0.1, -0.2, 0.3]]`:

```
i=0  t=9  c_skip=3.086324e-05 c_out=0.999985
     denoised=[[1.000015, -1.000046, 1.0], [0.999992, 1.000077, -1.000031]]
     prev=[[0.848282, -0.848293, 0.617787], [0.156807, 1.078794, -0.272066]]
i=1  t=5  c_skip=9.999000e-05 c_out=0.999950
i=2  t=2  c_skip=6.246096e-04 c_out=0.999688
     denoised=[[0.868255, -1.000937, 1.0], [0.233425, 1.001562, -1.000625]]
     prev equals denoised (the final step does not re-noise)
```

At `t=9`, \(s=90\) and \(c_{\text{skip}}=0.25/8100.25=3.086\times10^{-5}\); the same formula in EDM would additionally multiply by \(\sigma_d=0.5\), which is why \(c_{\text{out}}\) is nearly 1 at the noisy end here. On the final step `prev` is simply `denoised`, independent of the `noise` passed in.

## Constraints

- `sample` / `model_output` must be non-empty floating tensors of the same shape.
- `alphas_cumprod` must be a 1-D float tensor, non-empty, all finite, with every element in \((0,1)\).
- `timesteps` must be a 1-D **integer** non-empty tensor whose values all lie in \([0, T-1]\) (\(T\) = length of `alphas_cumprod`); duplicates and non-monotone order are allowed.
- `step_index` is an integer (`bool` does not count) in \([0, \text{len(timesteps)}-1]\).
- `prediction_type` accepts only `epsilon` / `sample` / `v_prediction`; `clip_sample` must be `bool`; `clip_sample_range` / `timestep_scaling` must be finite positive numbers.
- Every non-final step must be given a float `noise` shaped like `sample`, otherwise `ValueError`.
- Compute in float64 internally; the two tensor outputs are float32.
