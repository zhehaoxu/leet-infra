# Diffusion: One DDPM Reverse Posterior Step p(x_{t-1}|x_t,x_0)

## Problem

Sampling requires denoising \(x_t\) back to \(x_{t-1}\). If the model has already given a prediction of the clean sample \(\hat x_0\), then the true reverse posterior \(q(x_{t-1}\mid x_t, x_0)\) is Gaussian, with closed-form mean and variance (Ho et al. DDPM Eq. (7)–(11)):

\[ \mu_\theta = \frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,x_t + \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\,\hat x_0, \qquad \tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t \]

One sampling step \(x_{t-1} = \mu_\theta + \sqrt{\tilde\beta_t}\,z\), \(z\sim\mathcal N(0,I)\) (passed in explicitly to ensure determinism). Here \(\alpha_t=1-\beta_t\).

## Rules

Given `x_t`, `x0_pred` (the model-predicted \(\hat x_0\), same shape), the scalars `beta_t`, `alpha_bar_t` (\(\bar\alpha_t\)), `alpha_bar_prev` (\(\bar\alpha_{t-1}\)), and `z` (same shape), return `x_prev` (\(x_{t-1}\), same shape). float32. Ready-made diffusion libraries are forbidden.

## Examples

\[ x_t=\begin{bmatrix}0.4&-1.1\\ 0.9&2.2\end{bmatrix},\ \hat x_0=\begin{bmatrix}0.7&-1.5\\ 1.1&2.6\end{bmatrix},\ \beta_t=0.02,\ \bar\alpha_t=0.65,\ \bar\alpha_{t-1}=0.72,\ z=\begin{bmatrix}0.2&-0.3\\ 0.5&0.1\end{bmatrix} \]

```
x_prev = [[0.37602317333221436, -0.9818338751792908],
          [0.8293452262878418 ,  1.8810272216796875 ]]
```

## Constraints

- Scalars \(0<\bar\alpha_t\le\bar\alpha_{t-1}\le1\) (more signal toward the beginning), \(\beta_t\in(0,1)\); \(\alpha_t=1-\beta_t\).
- When \(z=0\), the output is exactly the posterior mean \(\mu_\theta\).
- The two coefficients satisfy: \(x_t\) weight \(\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\), \(\hat x_0\) weight \(\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}\), variance \(\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t\).
