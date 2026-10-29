# Discrete Diffusion: D3PM Uniform-Kernel Posterior Routes

## Problem Statement

Diffusion models over discrete state spaces (D3PM) do not output noise or velocity; they emit logits over the vocabulary at every position and sample a "clean token" \(x_0\) from them. Under the uniform corruption kernel, stepping from time \(t\) back to a cleaner \(s<t\) has a posterior \(q(x_s\mid x_t,x_0)\) made of three **mutually exclusive routes**: jump to the predicted clean token, stay put, or jump to a uniformly random token. One step of discrete DDIM is a single multinomial draw over those three routes.

This problem asks for the **deterministic core** of that step: given logits and the current tokens, produce the greedy denoised \(x_0\), the (normalized) route masses, and an optional leave-one-out (LOO) denoiser correction. The sampling itself is left to the RNG and is out of scope.

## Rules

Input `logits` has shape \((B, L, K)\) (per-position logits over \(K\) classes) and `tokens` is an integer tensor of shape \((B, L)\) (the current block tokens, each in \([0,K)\)). With \(N=\) `num_inference_steps`, the time accounting uses "alpha closer to 1 means cleaner":

\[ \alpha_t = \frac{\text{step\_index}}{N},\qquad \alpha_s = \frac{\text{step\_index}+1}{N},\qquad a = \frac{\alpha_t}{\alpha_s},\qquad b = \alpha_s \]

First take \(x_0=\arg\max\) of the logits. The **LOO correction** (`use_loo=True`) subtracts, from the logit at the observed token's position, before the argmax:

\[ \delta = \operatorname{log1p}\!\left(\frac{K\,\alpha_t}{1-\alpha_t}\right),\qquad \text{only when } 0<\alpha_t<1 \]

Write `same = (tokens == x0)`. The three route masses are

\[ m_{\text{clean}} = \frac{b\,(1-a)}{K} + a\,b\cdot\text{same},\qquad m_{\text{stay}} = \frac{a\,(1-b)}{K},\qquad m_{\text{noise}} = \frac{(1-a)(1-b)}{K} \]

**The triple is not always normalized**: it sums to \((1-ab)/K\) when `same` is false and to exactly 1 when true. Upstream samples from the triple after dividing by its own sum, so that division must be reproduced — it is not a no-op.

Return `(route_probs, pred_tokens)`:

- `route_probs`: float32, shape \((B, L, 3)\), the last dim ordered clean / stay / noise;
- `pred_tokens`: int64, shape \((B, L)\) — the argmax, possibly after the LOO correction.

## Example

\(B=1\), \(L=2\), \(K=4\), `logits` such that \(x_0=[2,1]\), `tokens=[[0,1]]`, `N=4`:

```
step_index=1 -> routes = [[0.333333, 0.333333, 0.333333], [0.714286, 0.142857, 0.142857]]
step_index=0 -> routes = [[0.250000, 0.000000, 0.750000], [0.250000, 0.000000, 0.750000]]
step_index=3 -> routes = [[1.000000, 0.000000, 0.000000], [1.000000, 0.000000, 0.000000]]
```

At `step_index=1`, \(\alpha_t=0.25,\alpha_s=0.5\): the first position has `same=False`, its three unnormalized masses are \(0.0625\) each and they normalize to a three-way tie; the second position has `same=True` and clean picks up the extra \(ab=0.25\). At `step_index=0` the stay mass is exactly zero; on the final step (`alpha_s=1`) only clean=1 survives, i.e. a deterministic commit.

The LOO example (\(K=4\), `logits=[[0,1,0,0]]`, `tokens=[[1]]`, `N=4`):

```
step_index=2          -> pred=1, routes = [[0.900000, 0.066667, 0.033333]]
step_index=2, use_loo -> pred=0, routes = [[0.500000, 0.333333, 0.166667]]
```

\(\delta=\log(5)\approx1.609\) pushes the observed token's logit from \(1.0\) to \(-0.609\), so the argmax flips to token 0, `same` turns false, and the clean mass collapses.

## Constraints

- `logits` must be a 3-D tensor and `tokens` a 2-D integer tensor whose shape equals the first two dims of `logits`; a floating-point `tokens` raises `ValueError`.
- `tokens` must lie in \([0,K)\), otherwise (including negatives) raise `ValueError`; \(K\ge 2\).
- `step_index` is an integer (`bool` does not count) with \(0\le \text{step\_index}<N\) and \(N\ge 1\).
- With `use_loo=True`, `step_index=0` (\(\alpha_t=0\)) applies no shift at all, so the result must equal `use_loo=False` exactly.
- Every row of `route_probs` sums to 1 (within float32 precision).
- Compute in float64 internally if you like; return `route_probs` as float32 and `pred_tokens` as int64.
