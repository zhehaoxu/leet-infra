# Diffusion: CFG Rescaling and the Guidance Interval

## Problem

Classifier-free guidance extrapolates between the unconditional and conditional predictions:

\[ \hat\epsilon = \epsilon_u + w\,(\epsilon_c-\epsilon_u) \]

With a large \(w\) the extrapolation leaves the training distribution and images oversaturate (high contrast, blown-out patches). **Guidance rescaling** (Lin et al.'s std matching) pulls it back by each sample's overall standard deviation:

\[ \hat\epsilon \leftarrow \hat\epsilon\cdot\frac{\operatorname{std}(\epsilon_c)}{\operatorname{std}(\hat\epsilon)} \]

Note this only rescales to the same magnitude and **does not preserve the mean**. The other issue is that guidance is not helpful at every step: the **guidance interval** (Kynkäänniemi et al.) applies guidance only for \(t\in[t_{\text{low}},t_{\text{high}}]\) and returns the conditional prediction outside it. This problem fuses both into one operator with a blend strength \(\varphi\): \(\varphi=0\) is bare extrapolation, \(\varphi=1\) is full rescaling.

## Rules

Given same-shaped `eps_cond` / `eps_uncond`, `guidance_scale=w`, the current step `t`, the interval `t_low` / `t_high` (inclusive), the strength `rescale=φ∈[0,1]` and the reduction dims `dims` (default all but dim 0, population standard deviation), return a float32 result. When `t` is outside the interval, return `eps_cond` unchanged.

## Examples

\[ \epsilon_c=\begin{bmatrix}0.4&-1.2&0.6\\0.0&0.5&-0.5\end{bmatrix},\ \epsilon_u=\begin{bmatrix}0.1&-0.4&0.2\\0.2&-0.1&0.0\end{bmatrix},\ w=3,\ t=0.5 \]

```
rescale=1 -> [[ 0.425607, -1.191700,  0.595850], [-0.123014,  0.522809, -0.461302]]
rescale=0 -> [[ 1.000000, -2.800000,  1.400000], [-0.400000,  1.700000, -1.500000]]
```

## Constraints

- `w` must be non-negative and finite; \(\varphi\in[0,1]\); `t`, `t_low` and `t_high` must lie in \([0,1]\) with `t_low<=t_high`.
- `dims` must be non-empty, must not contain dim 0 (the std is per sample), and must be in range without duplicates; the input is at least 2-D.
- **Degeneracies**: at \(w=1\), \(\hat\epsilon=\epsilon_c\) (the rescaling ratio is exactly 1, so the output equals `eps_cond`); outside the interval the output equals `eps_cond` as well.
- With \(\varphi=1\) each sample's population std equals that of `eps_cond`; with \(\varphi=0.5\) the result is the linear blend of bare extrapolation and the fully rescaled version.
- A zero std after guidance (e.g. both branches constant) cannot be rescaled and must raise `ValueError`.
