# Diffusion: Adaptive Projected Guidance

## Problem

Classifier-free guidance extrapolates along the difference of the two branch predictions:

\[ \delta=\epsilon_c-\epsilon_u \]

Decomposed relative to the conditional prediction \(\epsilon_c\), the component parallel to it mainly changes the magnitude (which is what blows up at large \(w\)), while the orthogonal component is what actually changes direction. **Adaptive Projected Guidance** (APG) rewrites guidance around that decomposition: keep the orthogonal part as is, damp the parallel part by a factor \(\eta\), and you retain the conditioning without pushing the magnitude off the rails.

This problem also carries the two engineering pieces APG uses inside samplers: an optional **momentum buffer** (an exponential moving average of consecutive differences) and a **norm clamp** (per-sample cap \(\tau\) on the L2 norm of the update vector).

## Rules

Given same-shaped `eps_cond` / `eps_uncond`, `guidance_scale=s`, the parallel retention factor `eta`, the norm threshold `norm_threshold=τ`, the momentum factor `momentum` with its previous buffer `prev_momentum`, and the reduction dims `dims`, apply in order:

1. \(\delta=\epsilon_c-\epsilon_u\); with momentum enabled, update the buffer \(b\leftarrow\delta+m\cdot b_{\text{prev}}\) (\(b_{\text{prev}}\) defaults to zeros) and set \(\delta=b\).
2. If \(\tau>0\), set \(\delta\leftarrow\delta\cdot\min(1,\tau/\lVert\delta\rVert_2)\), one scalar factor per sample.
3. Take \(\hat{e}=\epsilon_c/\max(\lVert\epsilon_c\rVert_2,10^{-12})\) and split
   \[ \delta_\parallel=(\delta\cdot\hat{e})\,\hat{e},\qquad \delta_\perp=\delta-\delta_\parallel \]
4. Combine \(\text{update}=\delta_\perp+\eta\,\delta_\parallel\) and extrapolate from the **unconditional** prediction:
   \[ \text{out}=\epsilon_u+s\cdot\text{update} \]

Return `(pred, new_momentum)`: `pred` is a float32 tensor, `new_momentum` the updated buffer (float32, same shape) or `None` when momentum is disabled. The buffer stores \(\delta\) **before** the norm clamp.

## Examples

\[ \epsilon_c=\begin{bmatrix}0.4&-1.2&0.6\\0.0&0.5&-0.5\end{bmatrix},\ \epsilon_u=\begin{bmatrix}0.1&-0.4&0.2\\0.2&-0.1&0.0\end{bmatrix},\ s=3,\ \tau=0 \]

```
eta=1.0 -> [[ 1.000000, -2.800000,  1.400000], [-0.400000,  1.700000, -1.500000]]
eta=0.5 -> [[ 0.595918, -1.587755,  0.793878], [-0.400000,  0.875000, -0.675000]]
eta=0.0 -> [[ 0.191837, -0.375510,  0.187755], [-0.400000,  0.050000,  0.150000]]
```

The `eta=1` row is bit-for-bit plain CFG; in the `eta=0` row the update vector is orthogonal to \(\epsilon_c\).

## Constraints

- Inputs are at least 2-D (dim 0 is the batch) and must share a shape; `s` is non-negative and finite, \(\eta\in[0,1]\), \(\tau\ge 0\), \(m\in[0,1)\).
- `dims` defaults to all dims but 0; it must be non-empty, exclude dim 0, and be in range without duplicates.
- `prev_momentum` may only be passed with momentum enabled and must match the input shape; without momentum the returned buffer is `None`.
- **Degeneracies**: at \(\eta=1\) the output is exactly \(\epsilon_u+s(\epsilon_c-\epsilon_u)\); at \(\eta=0\) the update is orthogonal to \(\epsilon_c\); \(\eta\) interpolates linearly between the two ends. When \(\lVert\epsilon_c\rVert=0\) the direction is the zero vector and the whole difference counts as orthogonal.
- With an all-zero difference the clamp factor is 1 (no \(0/0\), no NaN).
- The clamp only affects the current step and is never written into the momentum buffer.
