# YaRN / LongRoPE: Wavelength Ramp Blending and Attention Temperature

## Problem

RoPE's blind spot lies beyond the training length: the angle of the low-frequency (long-wavelength) dimensions sweeps through many multiples of \(2\pi\), a combination the model has never seen. **YaRN** (Peng et al., 2309.00071; LongRoPE uses the same recipe) does not apply a blanket position interpolation. Instead it splits the dimensions by **wavelength** into three bands: high-frequency dimensions (short wavelength, fast rotation) are **extrapolated** unchanged, low-frequency dimensions (long wavelength, slow rotation) are **interpolated** by \(1/\text{factor}\), and a linear ramp bridges the two. The two boundaries are recovered from a "number of rotations" formula — `beta_fast` sets the extrapolation-side boundary, `beta_slow` the interpolation-side one.

Interpolation squashes every angle, so attention logits shrink, the distribution sharpens and its entropy drops. YaRN therefore adds a temperature \(\gamma = 0.1\ln s + 1\) multiplied onto cos/sin.

This problem asks for one implementation that covers three modes: `yarn` (ramp blending + temperature), `linear` (the whole inv_freq divided by factor) and `dynamic_ntk` (the base rewritten from the current length). The latter two are YaRN's degenerate counterparts, and they verify that the two ends of the ramp really do degenerate into them.

## Rules

Let \(d=\text{head\_dim}\) (even, \(d\ge4\)), \(h=d/2\), \(i=0,\dots,h-1\), \(L=\text{original\_max\_position\_embeddings}\):

\[ \text{pos\_freqs}_i = \text{base}^{\,2i/d},\qquad \text{inv\_extrap}_i=\frac{1}{\text{pos\_freqs}_i},\qquad \text{inv\_interp}_i=\frac{\text{inv\_extrap}_i}{\text{factor}} \]

### yarn

**Boundaries.** First invert the rotation count into a dimension index:

\[ c(n)=\frac{d\cdot\ln\!\big(L/(n\cdot 2\pi)\big)}{2\ln\text{base}} \]

With `truncate=True`, take \(low=\lfloor c(\text{beta\_fast})\rfloor\) and \(high=\lceil c(\text{beta\_slow})\rceil\); with `False`, keep the floats. Then clamp \(low=\max(low,0)\), \(high=\min(high,d-1)\), and add \(10^{-3}\) to \(high\) when \(low=high\) so the divisor never vanishes.

**Ramp and blending.** The ramp argument only spans \(h\) entries, while the boundaries are derived on the \(d\) scale:

\[ \text{ramp}_i=\mathrm{clamp}\!\left(\frac{i-low}{high-low},0,1\right),\qquad \text{inv\_freq}_i=\text{inv\_interp}_i\cdot\text{ramp}_i+\text{inv\_extrap}_i\cdot(1-\text{ramp}_i) \]

So dimensions with \(i<low\) are fully extrapolated, those with \(i>high\) are fully interpolated, and the rest transition linearly. Both endpoints decrease with \(i\) while the weight moves toward the smaller `inv_interp`, so the whole `inv_freq` is monotonically non-increasing.

**Temperature.** With \(g(s,m)=1\) for \(s\le1\) and \(0.1\,m\ln s+1\) otherwise,

\[ \gamma=\begin{cases}\text{attention\_factor}, & \text{given explicitly}\\[2pt] g(\text{factor},\text{mscale})\,/\,g(\text{factor},\text{mscale\_all\_dim}), & \text{both mscale values given}\\[2pt] g(\text{factor},1), & \text{otherwise}\end{cases} \]

### linear / dynamic_ntk

\[ \text{linear}:\ \text{inv\_freq}=\text{inv\_interp},\quad \gamma=1 \]

\[ \text{dynamic\_ntk}:\ \ell=\max(\text{seq\_len}\ \text{or}\ L,\ L),\quad \text{base}'=\text{base}\cdot\Big(\frac{\text{factor}\cdot\ell}{L}-(\text{factor}-1)\Big)^{d/(d-2)},\quad \text{inv\_freq}_i=\text{base}'^{-2i/d} \]

Neither carries a temperature; passing `attention_factor` / `mscale` / `mscale_all_dim` in these modes raises `ValueError`.

### cos / sin

\[ \text{freqs}=\text{position\_ids}_{[\dots,\mathrm{None}]}\cdot\text{inv\_freq},\qquad \text{emb}=[\text{freqs}\ \Vert\ \text{freqs}] \]

\[ \cos=\cos(\text{emb})\cdot\gamma,\qquad \sin=\sin(\text{emb})\cdot\gamma \]

Return `(inv_freq, gamma, cos, sin)`: `inv_freq` has shape \((h,)\) in float32, `gamma` is a python float, `cos` / `sin` have shape \((B,S,d)\) in float32.

## Example

Take \(d=8\), `base=10000`, `factor=4`, \(L=2048\) and `position_ids=[[0,1,2,3]]`. Then \(c(32)=1.008\), \(c(1)=2.513\), hence \(low=1\) and \(high=3\), and only \(i=2\) sits on the ramp (\(\text{ramp}=0.5\)):

```
inv_freq    = [1.0, 0.1, 0.00625, 0.00025]          gamma = 1.138629
low = 1, high = 3    only i=2 sits on the ramp: 0.5*0.0025 + 0.5*0.01 = 0.00625
cos(pos=3)  = [-1.127235, 1.087774, 1.138429, 1.138629, ...]   last four repeat the first four
sin(pos=3)  = [0.160683, 0.336488, 0.021348, 0.000854, ...]   gamma multiplies cos/sin alike
```

\(i=0\) is fully extrapolated (\(1.0\)), \(i=1\) is fully extrapolated too (\(0.1\)), and \(i=3\) is fully interpolated (\(0.001/4=0.00025\)). The temperature is \(\gamma=0.1\ln4+1=1.138629\): the cos at position 0 equals \(\gamma\) rather than 1, which is the direct consequence of "using the amplitude as a temperature" — it is not a rotation, it rescales the logits by \(\gamma^2\) overall.

## Constraints

- `head_dim` must be an even integer \(\ge4\); `base` must be \(>1\); `factor` must be finite and positive; `original_max_position_embeddings` is an integer \(\ge2\) (`bool` is rejected).
- `beta_fast` / `beta_slow` must be finite and positive with **`beta_fast > beta_slow`**; `truncate` must be a real `bool`.
- `position_ids` must be a non-empty, non-negative 2-D integer tensor of shape \((B,S)\).
- `rope_type` accepts only `"yarn"` / `"linear"` / `"dynamic_ntk"`; `seq_len` is used only under `dynamic_ntk` and must be a positive integer when given (values below \(L\) are raised to \(L\)).
- \(\gamma\) multiplies **both** cos and sin: a rotation preserves the norm, so applying this cos/sin to any vector multiplies its length by \(\gamma\).
- Compute internally in float64 and return float32.
