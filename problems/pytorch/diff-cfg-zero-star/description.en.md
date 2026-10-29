# Diffusion: CFG-Zero* Zero-Init and Optimal Scale

## Problem

In the first few steps of flow-matching sampling the model output is still unreliable: both branch predictions are off the true velocity field, so strong guidance just amplifies the error and carries it through every later step. **CFG-Zero\*** (arXiv:2503.18886) fixes this with two changes:

1. **Zero initialization**: for the first \(k\) steps the guided result is set to zero (the sampler's update vanishes there);
2. **Optimal scale**: afterwards the unconditional prediction \(\epsilon_u\) is no longer used directly as the extrapolation base. Instead a scalar \(\alpha\) is found so that \(\alpha\epsilon_u\) is the least-squares best fit to \(\epsilon_c\), and CFG extrapolates from \(\alpha\epsilon_u\).

## Rules

Given same-shaped `eps_cond` / `eps_uncond`, `guidance_scale=s`, the current `step`, `zero_init_steps=k` and the stabilizer `eps`:

- if `step < k`, return an all-zero float32 tensor of the input shape;
- otherwise compute a per-sample scalar
  \[ \alpha=\frac{\langle\epsilon_c,\epsilon_u\rangle}{\lVert\epsilon_u\rVert_2^2+\varepsilon} \]
  set \(\tilde{\epsilon}_u=\alpha\epsilon_u\), and return
  \[ \text{out}=\tilde{\epsilon}_u+s\left(\epsilon_c-\tilde{\epsilon}_u\right) \]

The inner product and the squared norm reduce over every dim but 0 (one scalar per sample).

## Examples

\[ \epsilon_c=\begin{bmatrix}0.4&-1.2&0.6\\0.0&0.5&-0.5\end{bmatrix},\ \epsilon_u=\begin{bmatrix}0.1&-0.4&0.2\\0.2&-0.1&0.0\end{bmatrix},\ s=3,\ k=1 \]

```
step=0 -> [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]           (inside the zero-init window)
step=3 -> [[ 0.590476, -1.161905,  0.580952], [ 0.400000,  1.300000, -1.500000]]
```

## Constraints

- `step` and `zero_init_steps` must be non-negative integers (booleans do not count); `s` is non-negative and finite, `eps` positive and finite.
- Inputs are at least 2-D and must share a shape.
- **Degeneracies**: at \(s=1\) the output equals `eps_cond` (the extrapolation endpoint returns to the conditional branch); for orthogonal branches \(\alpha=0\) and the output collapses to \(s\epsilon_c\); for parallel branches \(\alpha\approx 1\) and the output returns to `eps_cond`.
- \(\alpha\) may be negative (branches pointing opposite ways); never clip it to be non-negative.
- \(\alpha\) makes the residual \(\epsilon_c-\alpha\epsilon_u\) orthogonal to \(\epsilon_u\) — that is the least-squares convention, not cosine similarity.
- \(\varepsilon\) is added to the squared norm in the denominator, so it only shrinks \(\lvert\alpha\rvert\) and never affects its sign.
