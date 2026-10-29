# Diffusion: LTX-2 Three-Term Delta Guidance

## Problem Statement

LTX-2 is a joint audio-video DiT. Besides regular **CFG**, its sampler stacks two more guidance passes through deliberately degraded models:

- **STG** (spatio-temporal guidance): one forward pass with a few transformer blocks perturbed, yielding a lower-quality yet still text-aligned conditional prediction; "clean minus perturbed" pushes the result away from the degraded distribution;
- **modality-isolation guidance**: one forward pass with A2V / V2A cross-attention disabled, yielding a conditional prediction that only knows its own modality; the same kind of difference suppresses cross-modal bleed.

All three are written in LTX-2 in one **delta form**: each is measured against the conditional prediction \(\epsilon_c\) and the terms simply add — unlike CFG, which shifts away from \(\epsilon_u\). This problem asks for the single-modality, single-step combination, including the window and per-term switches.

## Rules

Given four same-shaped predictions (dim 0 is the batch): `pred_cond` \(\epsilon_c\), `pred_uncond` \(\epsilon_u\), `pred_stg`, `pred_modality`, plus the scales `cfg_scale = s_c`, `stg_scale = s_t`, `modality_scale = s_m`, the current `step`, the total `num_inference_steps = N`, and the window `start / stop`.

All three share one **left-closed, right-open** window:

\[ \text{within} = \big[\lfloor \text{start}\cdot N\rfloor \le \text{step} < \lfloor \text{stop}\cdot N\rfloor\big] \]

Per-term switches:

\[ \text{cfg\_on} = \text{within} \wedge \lnot\text{isclose}(s_c, 1.0) \]

\[ \text{stg\_on} = \text{within} \wedge \lnot\text{isclose}(s_t, 0.0) \]

\[ \text{mod\_on} = \text{within} \wedge \lnot\text{isclose}(s_m, 1.0) \]

The combination (each term appears only when it is on):

\[ \epsilon = \epsilon_c + (s_c-1)\,(\epsilon_c-\epsilon_u) + s_t\,(\epsilon_c-\epsilon_{stg}) + (s_m-1)\,(\epsilon_c-\epsilon_{mod}) \]

Note the scale semantics are **not uniform**: CFG and modality isolation are "identity at 1", so their increments subtract 1; STG is "identity at 0", so its increment coefficient is \(s_t\) itself. Writing \(s_c(\epsilon_c-\epsilon_u)\) conjures an extra guidance term at \(s_c=1\), and writing \((s_t-1)\) subtracts a difference at \(s_t=0\).

Return a float32 tensor of the same shape as `pred_cond`. A disabled term may be passed `None`; a term that is on but missing raises `ValueError`.

## Example

\[ \epsilon_c=\begin{bmatrix}0.4&-1.2&0.6\\0.0&0.5&-0.5\end{bmatrix},\ \epsilon_u=\begin{bmatrix}0.1&-0.4&0.2\\0.2&-0.1&0.0\end{bmatrix},\ \epsilon_{stg}=\begin{bmatrix}0.2&-0.8&0.5\\0.1&0.3&-0.2\end{bmatrix},\ \epsilon_{mod}=\begin{bmatrix}0.3&-1.0&0.55\\0.05&0.2&-0.3\end{bmatrix} \]

With `step=1`, `N=10` (the whole window \([0,10)\) is open):

```
default (1.0, 0.0, 1.0) -> [[ 0.400000, -1.200000,  0.600000], [ 0.000000,  0.500000, -0.500000]]
(3.0, 1.0, 1.5)         -> [[ 1.250000, -3.300000,  1.525000], [-0.525000,  2.050000, -1.900000]]
CFG only (3.0)          -> [[ 1.000000, -2.800000,  1.400000], [-0.400000,  1.700000, -1.500000]]
```

The first row uses the default scales: all three terms sit at their identity points, so the output is bit-identical to \(\epsilon_c\).

## Constraints

- All four inputs have at least one dimension (dim 0 is the batch) and identical shapes; a non-tensor input or a shape mismatch raises `ValueError`.
- `cfg_scale`, `stg_scale`, `modality_scale` are finite reals (`stg_scale` may be negative for reversed guidance); a non-real or non-finite value raises `ValueError`.
- `step` is an integer (`bool` does not count) with \(0\le \text{step} < N\) and \(N>0\).
- `start` and `stop` lie in \([0,1]\) with `start <= stop`.
- **Degenerate cases**: when every scale is at its identity point the output must be bit-identical to `pred_cond`; an empty window also returns `pred_cond` alone and requires no branch predictions at all.
- **Additivity**: all three deltas are measured against `pred_cond`, so "full result minus `pred_cond`" equals the sum of the three deltas.
- Compute in float64 internally if you like, but return float32.
