# Diffusion: DPM-Solver++(2M) Second-Order Update with Euler Base

## Problem

DPM-Solver++ (Lu et al.) writes the denoising ODE as an exponential integrator in **data-prediction** (x0-prediction) form: with \(\alpha_t=\sqrt{\bar\alpha_t}\), \(\sigma_t=\sqrt{1-\bar\alpha_t}\) and \(\lambda_t=\log(\alpha_t/\sigma_t)\) (half the log-SNR), one step from \(\bar\alpha_{s_0}\) to \(\bar\alpha_t\) is

\[ x_t = \frac{\sigma_t}{\sigma_{s_0}}\,x_{s_0} - \alpha_t\left(e^{-h}-1\right)\left(D_0 + \tfrac{1}{2}D_1\right),\qquad h=\lambda_t-\lambda_{s_0} \]

where \(D_0\) is the current x0 prediction. The 2M (multistep, second-order) variant extrapolates a first difference from the previous prediction:

\[ D_1 = \frac{D_0 - D_0^{\text{prev}}}{r_0},\qquad r_0=\lambda_{s_0}-\lambda_{s_1} \]

With `order=1` the \(D_1\) term is dropped. The first-order step is mathematically identical to the deterministic DDIM closed form — the most valuable takeaway here: two very different-looking formulas are the same exponential integrator.

## Rules

Given `x` (the current sample \(x_{s_0}\)), `x0_pred` (\(D_0\)), the scalars `alpha_bar_t` and `alpha_bar_s0`, plus `order`; for `order=2` also `x0_pred_prev` (\(D_0^{\text{prev}}\)) and `alpha_bar_s1`. Return a float32 tensor shaped like `x` (compute internally in float64). Ready-made samplers are forbidden.

## Examples

\[ x=1.0,\ D_0=0.5,\ D_0^{\text{prev}}=0.4,\ \bar\alpha_t=0.81,\ \bar\alpha_{s_0}=0.64,\ \bar\alpha_{s_1}=0.49 \]

```
order=1 -> 0.885890
order=2 -> 0.937698
```

## Constraints

- Every \(\bar\alpha\) must lie strictly inside \((0,1)\); `order` is 1 or 2 only.
- `order=2` requires \(r_0>0\), i.e. \(s_1\) must be one level noisier than \(s_0\) (\(\bar\alpha_{s_1}<\bar\alpha_{s_0}\)); otherwise raise `ValueError`.
- **Degeneracies**: \(\bar\alpha_t=\bar\alpha_{s_0}\) returns the input unchanged; \(\bar\alpha_t\to1\) converges to \(D_0\); when the two predictions coincide the second-order step collapses to first order.
- A single step never overshoots the x0 prediction (both the coefficient and the \(\sigma\) ratio are positive, so the result lies between \(x\) and \(D_0\)).
