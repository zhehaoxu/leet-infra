# VE/EDM Preconditioning: Four Coefficients and the Churned Euler Step

## Problem statement

EDM (Karras et al., 2206.00364) parameterises a diffusion network so that it works at any noise level \(\sigma\), instead of feeding it samples whose scale varies by orders of magnitude. `EDMEulerScheduler` exposes four coefficients: `c_in` and `c_noise` on the input side, `c_skip` and `c_out` on the output side. The network emits a raw output \(F\) in preconditioned space, which is recombined into a denoised sample by

\[ D(x;\sigma) = c_{\text{skip}}(\sigma)\,x + c_{\text{out}}(\sigma)\,F\big(c_{\text{in}}(\sigma)\,x;\ c_{\text{noise}}(\sigma)\big) \]

Sampling is then an Euler discretisation of that \(x_0\) prediction, with an optional stochastic "churn": raise the noise level to \(\hat\sigma\), add a little noise, and take the ODE step from there.

Implement the full "preconditioning plus single step" computation: return the model input, the conditioning scalar, the denoised sample, the next sample, and the noise level actually used.

## Rules

Let \(\sigma = \text{sigmas}[i]\), \(\sigma_d = \text{sigma\_data}\) and \(K = \text{len(sigmas)} - 1\).

**1. Churn**: only when \(\text{s\_tmin} \le \sigma \le \text{s\_tmax}\)

\[ \gamma = \min\!\left(\frac{\text{s\_churn}}{K},\ \sqrt{2}-1\right),\qquad \hat\sigma = \sigma(1+\gamma) \]

Outside that window \(\gamma = 0\) and \(\hat\sigma = \sigma\). When \(\gamma>0\), noise is added first:

\[ x \leftarrow x + \text{noise}\cdot \text{s\_noise}\cdot\sqrt{\hat\sigma^2-\sigma^2} \]

Here `noise` is an **explicit argument** (a floating-point tensor shaped like the sample); when \(\gamma=0\) it takes no part in the computation at all.

**2. Input-side coefficients** (using the **pre-churn** \(\sigma\)):

\[ c_{\text{in}} = \frac{1}{\sqrt{\sigma^2+\sigma_d^2}},\qquad c_{\text{noise}} = \tfrac14\ln\sigma,\qquad \text{model\_input} = x_{\text{original}}\cdot c_{\text{in}} \]

**3. Output-side coefficients** (using the **post-churn** \(\hat\sigma\)):

\[ c_{\text{skip}} = \frac{\sigma_d^2}{\hat\sigma^2+\sigma_d^2},\qquad c_{\text{out}} = \pm\frac{\hat\sigma\,\sigma_d}{\sqrt{\hat\sigma^2+\sigma_d^2}} \]

`prediction_type="epsilon"` takes the plus sign and `"v_prediction"` the minus sign.

\[ \text{denoised} = c_{\text{skip}}\cdot x + c_{\text{out}}\cdot \text{model\_output} \]

**4. Euler step**:

\[ \text{derivative} = \frac{x-\text{denoised}}{\hat\sigma},\qquad dt = \text{sigmas}[i+1] - \hat\sigma,\qquad \text{prev\_sample} = x + \text{derivative}\cdot dt \]

Return `(model_input, c_noise, prev_sample, denoised, sigma_hat)`: the first three are float32 tensors shaped like `sample`, while `c_noise` and `sigma_hat` are Python floats.

## Example

With `sample=[[1.0, -2.0]]`, `model_output=[[0.5, 0.25]]`, `sigmas=[2.0, 1.0, 0.5, 0.0]` and `sigma_data=0.5`:

```
i=0            -> model_input=[0.485071, -0.970142]  c_noise=0.173287  sigma_hat=2.0
                 denoised=[0.301359, 0.003621]       prev=[0.65068, -0.99819]
i=1            -> c_noise=0.0                        denoised=[0.423607, -0.288197]
i=2            -> c_noise=-0.173287                  denoised=[0.676777, -0.911612]  prev equals denoised
s_churn=10     -> sigma_hat=2.828427                 model_input stays [0.485071, -0.970142]
                 denoised=[0.282547, 0.044304]       prev=[0.606915, -0.890599]
```

At step 0, \(c_{\text{in}} = 1/\sqrt{4.25} = 0.485071\), which happens to equal \(c_{\text{out}}\) when \(\sigma_d=0.5\) — not a coincidence, since the \(\epsilon\) branch gives \(c_{\text{out}} = c_{\text{in}}\cdot\sigma\sigma_d\) and \(\sigma\sigma_d=1\). Step 2 is the last step, where \(\text{sigmas}[3]=0\) makes \(dt=-\hat\sigma\), so `prev_sample` equals `denoised` exactly. With `s_churn=10`, \(\gamma\) is clamped at \(\sqrt2-1\) so \(\hat\sigma = 2\sqrt2\); note that `model_input` still uses the original sample and \(\sigma=2\), while only `denoised`/`prev` use the churned sample and \(\hat\sigma\).

## Constraints

- `sample` and `model_output` must be non-empty floating-point tensors of the same shape; `sigmas` must be a 1-D floating-point tensor of length at least 2, finite, non-negative and non-increasing.
- `step_index` is an integer (`bool` does not count) in \([0, K-1]\); the current step's \(\sigma\) must be positive.
- `sigma_data` must be a finite positive number; `s_churn` and `s_noise` must be finite and non-negative; `s_tmin \le s_tmax`, with `s_tmax` defaulting to \(+\infty\).
- `prediction_type` accepts only `"epsilon"` / `"v_prediction"`; anything else raises `ValueError`.
- When \(\gamma>0\) a floating-point `noise` shaped like `sample` must be supplied explicitly, otherwise `ValueError` is raised; when \(\gamma=0\) `noise` is ignored.
- Compute in float64 internally and return the three tensors as float32.
