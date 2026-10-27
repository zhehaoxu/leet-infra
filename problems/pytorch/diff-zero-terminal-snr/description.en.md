# Diffusion: Zero-Terminal-SNR Rescaling and log-SNR Parameterization

## Problem

For the usual schedule the final cumulative product \(\bar\alpha_T=\prod(1-\beta_i)\) is still on the order of \(10^{-4}\), so the terminal \(\mathrm{SNR}=\bar\alpha/(1-\bar\alpha)\) is far from zero: the model never sees "pure noise" and the first inference step lands in a region training never covered. Zero-terminal-SNR (Lin et al.) rescales the whole schedule so the endpoint is exactly \(\bar\alpha'_T=0\):

\[ \bar\alpha'_t=\frac{\bar\alpha_t-\bar\alpha_T}{1-\bar\alpha_T} \;\Longrightarrow\; 1-\bar\alpha'_t=\frac{1-\bar\alpha_t}{1-\bar\alpha_T} \]

i.e. \((1-\bar\alpha)\) is uniformly scaled by \(1/(1-\bar\alpha_T)\), giving \(\bar\alpha'_0=1\) (when \(\beta_0=0\)) and an exact \(\bar\alpha'_T=0\). Everything afterwards is accounted for in log-SNR space:

\[ \lambda_t=\log\frac{\bar\alpha'_t}{1-\bar\alpha'_t}\in(-\infty,+\infty],\qquad \theta_t=\operatorname{atan2}\!\left(\sqrt{1-\bar\alpha'_t},\sqrt{\bar\alpha'_t}\right) \]

\(\theta\) is the rotation angle from \(\epsilon\) to \(v\): \(v=\cos\theta\,\epsilon-\sin\theta\,x_0\). At the terminal \(\theta=\pi/2\) we get \(v=-x_0\) with no division by \(\alpha=0\) — exactly why zero-terminal-SNR has to be paired with v-prediction.

## Rules

Given a 1-D `betas` (entries in \([0,1)\), index 0 cleanest and the last entry noisiest), return `(alpha_bars, log_snrs, angles)` as three same-length float32 tensors: the rescaled \(\bar\alpha'\), \(\lambda\) and \(\theta\). The endpoints use \(\lambda=-\infty\) (for \(\bar\alpha'=0\)) and \(+\infty\) (for \(\bar\alpha'=1\)), with angles \(\pi/2\) and \(0\).

## Examples

\[ \text{betas}=[0.1,0.2,0.4]\Rightarrow\bar\alpha=[0.9,0.72,0.432],\ \bar\alpha_T=0.432 \]

```
alpha_bars = [0.823944, 0.507042, 0.0]
log_snrs   = [1.543298, 0.028171, -inf]
angles     = [0.432994, 0.778356, 1.570796]
```

## Constraints

- Every entry of `betas` must lie in \([0,1)\) with no nan/inf; an empty or non-1-D tensor raises `ValueError`.
- An all-zero `betas` makes \(\bar\alpha_T=1\), turning the rescaling into \(0/0\); it must raise `ValueError`.
- **Degeneracies**: \(\bar\alpha'\) is non-increasing, \(\lambda\) is non-increasing and \(\theta\) is non-decreasing; the terminal \(\bar\alpha'\) is exactly 0 (not \(10^{-7}\)) while the one before it is still positive.
- \(\theta\) and \(\lambda\) satisfy the closed form \(\theta=\arctan(e^{-\lambda/2})\), with \(\cos\theta=\alpha\) and \(\sin\theta=\sigma\).
