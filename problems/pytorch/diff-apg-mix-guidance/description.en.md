# Diffusion: APG-Mix CFG Warmup and Orthogonal Projected Guidance

## Problem Statement

**Adaptive Projected Guidance** (APG) splits the CFG direction \(\Delta=\epsilon_c-\epsilon_u\) into a component along the conditional prediction \(\epsilon_c\) and a component orthogonal to it. It keeps only the orthogonal part (optionally mixing in a fraction of the parallel part via `eta`) before scaling, which avoids pushing the sample along the magnitude of \(\epsilon_c\) at high scales — the usual cause of overexposure and saturation.

APG-Mix is the combined recipe used by HunyuanImage-2.1: **run plain CFG for the first few denoising steps while warming up a momentum buffer with the CFG difference, then switch to APG** from some step on. The guidance direction is then no longer the current difference but the momentum-smoothed one. This problem asks for that single-step operator, with the momentum state carried out explicitly across steps.

## Rules

Given same-shaped `pred_cond` \(\epsilon_c\) and `pred_uncond` \(\epsilon_u\) (dim 0 is the batch), the current `step`, the total `num_inference_steps = N`, a cross-step momentum state `buffer` (same-shaped tensor, `None` on the first call), and the parameters `cfg_scale = s_c`, `apg_scale = s_a`, `momentum = \mu`, `norm_threshold = \tau`, `eta`, `apg_start_step`, the window `start / stop`, plus `use_original_formulation`.

First decide the two switches:

\[ \text{cfg\_on} = \big[\lfloor \text{start}\cdot N\rfloor \le \text{step} < \lfloor \text{stop}\cdot N\rfloor\big] \ \wedge\ \lnot\text{isclose}(s_c,\, \text{off}) \]

\[ \text{apg\_on} = \text{cfg\_on} \ \wedge\ \text{step} > \text{apg\_start\_step} \ \wedge\ \lnot\text{isclose}(s_a,\, \text{off}) \]

where \(\text{off}=1.0\) by default or \(0.0\) when `use_original_formulation=True`, and `isclose` uses `math.isclose`'s default tolerances. Note APG's start test is **strictly greater**: at `step == apg_start_step` plain CFG still runs. APG depends on CFG, so it can never be on while CFG is off.

With \(d=\epsilon_c-\epsilon_u\), the momentum update is

\[ \text{running} = d + \mu\cdot\text{running\_prev},\qquad \text{running\_prev}=0\ \text{(when buffer is None)} \]

`momentum=None` disables momentum entirely (then `buffer` must be `None`) and the current-step \(d\) is always used.

The three branches:

| case | output | momentum |
| --- | --- | --- |
| `cfg_on` false | \(\epsilon_c\) | untouched, `buffer` returned as is |
| CFG only | \(\text{base} + s_c\, d\) | updated to `running` |
| APG | \(\text{base} + s_a\,\big(\text{orth} + \eta\,\text{parallel}\big)\) | updated to `running` (pre-clamp) |

Here \(\text{base}=\epsilon_u\) by default or \(\epsilon_c\) when `use_original_formulation=True`. The APG branch decomposes the **clamped** difference \(\tilde{d}\):

\[ \tilde{d} = \begin{cases} d_{\text{used}}\cdot\min\!\left(1,\ \tau/\lVert d_{\text{used}}\rVert_2\right), & \tau>0\\ d_{\text{used}}, & \tau=0\end{cases} \]

\[ u = \frac{\epsilon_c}{\lVert \epsilon_c\rVert_2},\qquad \text{parallel}=\langle \tilde{d}, u\rangle\, u,\qquad \text{orth}=\tilde{d}-\text{parallel} \]

\(d_{\text{used}}\) is the `running` value above (smoothed when momentum is on, the current difference otherwise). Every norm and inner product is per-sample over **all dimensions except dim 0 (batch)**, and the \(\lVert\cdot\rVert_2\) denominator is floored at \(10^{-12}\).

Two asymmetries matter: (1) the CFG branch shifts by the **unsmoothed** \(d\); momentum is merely written to the buffer on the side; (2) the buffer stores the **pre-clamp** `running`, so the clamp only affects the current step's output.

Return `(pred, new_buffer)`, both float32; `new_buffer` is same-shaped as `pred_cond` when momentum is on, and `None` when `momentum=None`. The input `buffer` must not be mutated in place.

## Example

\[ \epsilon_c=\begin{bmatrix}0.4&-1.2&0.6\\0.0&0.5&-0.5\end{bmatrix},\quad \epsilon_u=\begin{bmatrix}0.1&-0.4&0.2\\0.2&-0.1&0.0\end{bmatrix} \]

With default parameters (\(s_c=3.5\), \(s_a=10\), \(\mu=-0.5\), \(\tau=10\), \(\eta=0\), `apg_start_step=5`, window \([0,10)\)):

```
step=3  pred = [[ 1.150000, -3.200000,  1.600000], [-0.500000,  2.000000, -1.750000]]
step=3  buf  = [[ 0.300000, -0.800000,  0.400000], [-0.200000,  0.600000, -0.500000]]
step=6  pred = [[ 0.253061, -0.359184,  0.179592], [-0.800000,  0.150000,  0.250000]]
step=6  buf  = [[ 0.150000, -0.400000,  0.200000], [-0.100000,  0.300000, -0.250000]]
step=6* pred = [[ 0.406122, -0.318367,  0.159184], [-1.800000,  0.400000,  0.500000]]
```

The starred row is the same step with `buffer=None` (starting from zero): its momentum equals the current difference, so the projected direction differs. The `step=6` rows take the `buf` returned at `step=3`, so the momentum is \(d_6-0.5\,d_3=0.5\,d_6\).

## Constraints

- `pred_cond` is required; `pred_uncond` is required whenever `cfg_on` holds. Both must have at least one dimension and identical shapes, else raise `ValueError`.
- `step` is an integer (`bool` does not count) with \(0\le \text{step} < N\) and \(N>0\); `apg_start_step` is a non-negative integer.
- `cfg_scale`, `apg_scale`, `norm_threshold`, `eta` are finite non-negative numbers; `momentum` is `None` or a finite real (possibly negative).
- `buffer` is `None` or a same-shaped tensor; passing a `buffer` together with `momentum=None` raises `ValueError`.
- `start` and `stop` lie in \([0,1]\) with `start <= stop`.
- **Degenerate cases**: with `momentum=None` the returned buffer is always `None`; with \(\eta=1\) and \(\tau=0\) APG degenerates to plain CFG on `base`; with \(\eta=0\) the output delta is per-sample orthogonal to \(\epsilon_c\).
- Compute in float64 internally if you like, but return float32.
