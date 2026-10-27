# In Depth: Zero-Terminal-SNR Rescaling and log-SNR Parameterization

## Origin

Lin et al., *Common Diffusion Noise Schedules and Sample Steps are Flawed* (arXiv:2305.08891) §3.1: the terminal SNR of the training schedule is not zero and the last sampling step still leaves noise, which breaks consistency; the paper gives the rescaling and requires v-prediction alongside it. diffusers' `rescale_betas_zero_snr=True` plus `prediction_type="v_prediction"` are exactly these two steps.

## Core points and pitfalls

1. **The rescaling scales only \(1-\bar\alpha\)**: \(\bar\alpha'=(\bar\alpha-\bar\alpha_T)/(1-\bar\alpha_T)\) is not "scale \(\bar\alpha\) proportionally"; only the former keeps \(\bar\alpha'_0=1\).
2. **The terminal value must be exactly 0**: the judge asserts `float(alpha_bars[-1]) == 0.0`. Approximating with `1 - 1e-7` breaks the next step — \(\lambda_T\) has to be a true \(-\infty\).
3. **±inf cannot come from one uniform formula**: \(\lambda=\log(\bar\alpha/(1-\bar\alpha))\) is \(-\infty\)/\(+\infty\) at the ends, so the implementation must branch and still write those into a float32 tensor (`torch.tensor(float('-inf'))` is a legal value).
4. **Why v-prediction is mandatory**: at the terminal \(\alpha=0\), the ε-parameterized recovery \(x_0=(x_t-\sigma\epsilon)/\alpha\) divides by zero, while the v-parameterized \(x_0=\alpha x_t-\sigma v\) stays defined (collapsing to \(-\sigma v\)). That is precisely \(\theta_T=\pi/2\), \(v=-x_0\).
5. **Uniform β has a closed form**: \(\bar\alpha'_t=((1-\beta)^{t+1}-(1-\beta)^n)/(1-(1-\beta)^n)\), which directly verifies that the rescaling did not corrupt the ratios.

## Verification approach

The reference solution runs `cumprod` then rescales (float64) and computes \(\lambda\) and \(\theta\) per entry with `math`. The judge's `_oracle` rolls the cumulative product in a python loop (no `cumprod`) and recomputes everything, using the `atan` form instead of `atan2`; it adds five structural groups: the three terminal zero-SNR facts (ᾱ=0, λ=-inf, θ=90°), the three clean-end facts for \(\beta_0=0\) (ᾱ=1, λ=+inf, θ=0), the \(\cos\theta/\sin\theta\) versus \(\alpha/\sigma\) consistency, the three monotonicities, and \(\theta=\arctan(e^{-\lambda/2})\) plus the uniform-β closed form. Tolerance `allclose(atol=1e-6)`.

## Complexity

\(O(T)\): one cumulative product and two element-wise transcendental passes.
