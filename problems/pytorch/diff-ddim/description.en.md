# Diffusion: One Deterministic DDIM Denoising Step

## Problem

DDIM (Song et al., arXiv:2010.02502) turns DDPM's stochastic sampling into a **deterministic** non-Markovian reverse process: with \(\sigma_t=0\), no noise is injected under the same \(\hat x_0\) prediction, so fewer steps suffice and a deterministic result can be "inverted" from the same noise.

Given the model-predicted noise \(\epsilon_\theta\), first invert it to obtain the clean-sample estimate:

\[ \hat x_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta}{\sqrt{\bar\alpha_t}} \]

Then jump directly to \(t-1\) (the DDIM update with \(\sigma_t=0\), Song's Eq. (12)):

\[ x_{t-1} = \sqrt{\bar\alpha_{t-1}}\,\hat x_0 + \sqrt{1-\bar\alpha_{t-1}}\,\epsilon_\theta \]

## Rules

Given `x_t`, `eps_pred` (\(\epsilon_\theta\), same shape), and the scalars `alpha_bar_t` (\(\bar\alpha_t\)) and `alpha_bar_prev` (\(\bar\alpha_{t-1}\)), return `x_prev` (same shape). float32, with no stochastic term. Ready-made diffusion libraries are forbidden.

## Examples

\[ x_t=\begin{bmatrix}0.4&-1.1\\ 0.9&2.2\end{bmatrix},\ \epsilon_\theta=\begin{bmatrix}0.3&-0.7\\ 1.2&-0.4\end{bmatrix},\ \bar\alpha_t=0.65,\ \bar\alpha_{t-1}=0.72 \]

```
x_prev = [[0.39293810725212097, -1.0922671556472778],
          [0.8350236415863037 ,  2.352832794189453 ]]
```

## Constraints

- Scalars \(0<\bar\alpha_t\le\bar\alpha_{t-1}\le1\).
- **Degenerate properties**: when \(\bar\alpha_{t-1}=\bar\alpha_t\), the update is the identity (\(x_{t-1}=x_t\)); when \(\bar\alpha_{t-1}=1\), \(x_{t-1}=\hat x_0\) (a single step fully denoises to the clean estimate).
- If `x_t` was generated from some \(x_0\) by the forward formula with the same `eps_pred`, then \(\hat x_0\) recovers that \(x_0\) exactly.
