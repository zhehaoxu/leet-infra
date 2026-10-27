# Diffusion: Timestep Shift for Rectified-Flow Sampling

## Problem

RF / Flow Matching samplers use a uniform step grid, but high-resolution samples (many tokens) need more of their steps spent in the **high-noise regime** — SD3 and Flux push the whole time grid toward the noise side with a one-parameter map:

\[ t' = \frac{s\,t}{1+(s-1)\,t} \;=\; \sigma\!\left(\operatorname{logit}(t)+\mu\right),\qquad s=e^{\mu} \]

that is, a shift by \(\mu\) in the logit domain. \(\mu\) itself is linearly interpolated from the image sequence length (higher resolution, longer sequence, larger \(\mu\)):

\[ \mu = m\cdot\text{seq\_len}+b,\qquad m=\frac{\mu_{\max}-\mu_{\text{base}}}{\text{len}_{\max}-\text{len}_{\text{base}}},\qquad b=\mu_{\text{base}}-m\cdot\text{len}_{\text{base}} \]

## Rules

Given a 1-D `times` and exactly one of: `shift` (the value \(s\) directly), `mu` (with \(s=e^{\mu}\) derived internally), or `seq_len` (an integer interpolated to \(\mu\) as above). The remaining parameters `base_shift=0.5`, `max_shift=1.15`, `base_seq_len=256`, `max_seq_len=4096` are overridable. Return `(shifted, s)`: a float32 tensor shaped like `times`, and a python float.

## Examples

\[ \text{seq\_len}=256\Rightarrow\mu=0.5,\ s=e^{0.5}=1.6487213,\ \text{times}=[0.25,0.5,0.75] \]

```
shifted = [0.354661, 0.622459, 0.831824]
s       = 1.6487212707
```

## Constraints

- `times` must be a 1-D float tensor whose entries all lie in \([0,1]\) with no nan / inf.
- Exactly **one** of `shift` / `mu` / `seq_len` is given; `shift` must be a positive finite real and `seq_len` a non-negative integer.
- **Degeneracies**: \(s=1\) (i.e. \(\mu=0\)) is the identity; the endpoints are fixed (\(t=0\to0\), \(t=1\to1\)); the map is strictly increasing.
- **Composition**: shifting by \(s_1\) and then \(s_2\) equals a single shift by \(s_1s_2\) (logit-domain shifts add).
- \(s>1\) pushes time toward the noise side (\(t'\ge t\)) and \(s<1\) toward the clean side; below `base_seq_len` the linear interpolation extrapolates to a smaller \(s\) instead of clamping.
