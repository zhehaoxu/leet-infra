# Diffusion: AutoGuidance Flips Its Anchor and Its Identity Threshold Together

## Problem Statement

AutoGuidance (*Guiding a Diffusion Model with a Bad Version of Itself*) does not train a separate null-condition model. Instead it **degrades the same conditional model** — dropping a few transformer blocks with some probability during sampling — and pushes the strong prediction away from that weak one, amplifying the gap between the two distributions. At the code level it still reuses the CFG two-prediction signature: `pred_cond` is the full model's prediction and `pred_uncond` carries the degraded model's prediction.

diffusers' `AutoGuidance` has two coupled switches. The first is `use_original_formulation`: it moves **both the extrapolation anchor and the identity scale** (the value at which guidance switches itself off). The default implementation anchors on the weak prediction and treats \(s=1\) as identity; the original-paper formulation anchors on the conditional prediction and treats \(s=0\) as identity. The second is `math.isclose`: compared against \(1.0\) it uses a **relative** tolerance (`rel_tol=1e-9` by default), while compared against \(0.0\) only an **absolute** tolerance can bite (`abs_tol=0` by default). So "close enough counts as identity" holds in the default formulation only.

Implement this single-step operator, including the left-closed right-open integer window and the optional standard-deviation pull-back (`guidance_rescale`).

## Rules

Two same-shape predictions with batch in dimension 0: `pred_cond` \(\epsilon_c\) and `pred_uncond` \(\epsilon_w\). The scale \(s\) is `guidance_scale`, `step` is the current index, \(N\) is `num_inference_steps`, `start / stop` give the window, `use_original_formulation` picks the formulation, and \(\phi\) is `guidance_rescale`.

Guidance is on only when both conditions hold:

\[ \text{within} = \big[\lfloor \text{start}\cdot N\rfloor \le \text{step} < \lfloor \text{stop}\cdot N\rfloor\big] \]

\[ \text{close} = \begin{cases}\text{isclose}(s, 0.0) & \text{use\_original\_formulation}\\[2pt]\text{isclose}(s, 1.0) & \text{otherwise}\end{cases}\qquad \text{on} = \text{within}\wedge\neg\text{close} \]

The guidance itself:

\[ \text{pred} = \begin{cases}\epsilon_c & \text{off}\\[3pt] \epsilon_c + s\,(\epsilon_c-\epsilon_w) & \text{on, original formulation}\\[3pt] \epsilon_w + s\,(\epsilon_c-\epsilon_w) & \text{on, default formulation}\end{cases} \]

Pull-back (runs whenever \(\phi>0\), **including the off branch**): per sample, over every dimension except dimension 0, using the unbiased standard deviation (`torch.std` default, dividing by \(n-1\)),

\[ \text{pred} \leftarrow \phi\cdot \text{pred}\cdot\frac{\operatorname{std}(\epsilon_c)}{\operatorname{std}(\text{pred})} + (1-\phi)\cdot \text{pred} \]

When guidance is off, \(\epsilon_w\) takes part in nothing — it may be `None` (even a tensor of the wrong shape). Passing `None` while guidance is on raises `ValueError`. Return a float32 tensor with the same shape as `pred_cond`.

Three traps: under the default formulation \(s=0\) is **not** identity — it moves the result entirely onto the weak prediction \(\epsilon_w\); under the original formulation \(s=1\) is not identity either — it extrapolates one more `shift`; and \(s=1+10^{-10}\) gets classified as identity (hence off) in the default formulation while \(s=10^{-12}\) stays on in the original one.

## Example

```
eps_c = [[1.0, 2.0, 3.0, 4.0], [0.0, 1.0, -1.0, 2.0]]
eps_w = [[0.2, 1.0, 2.5, 3.0], [0.5, 0.0, 0.5, -0.5]]
step = 5, N = 20, start = 0.1, stop = 0.8   # window = [2, 16), step 5 is inside
```

With `phi = 0` (no pull-back):

```
s=7.5 native            -> [[ 6.200000,  8.500000,   6.250000, 10.500000], [-3.250000,  7.500000, -10.750000, 18.250000]]
s=7.5 original          -> [[ 7.000000,  9.500000,   6.750000, 11.500000], [-3.750000,  8.500000, -12.250000, 20.750000]]
s=1.0 native (off)      -> [[ 1.000000,  2.000000,   3.000000,  4.000000], [ 0.000000,  1.000000,  -1.000000,  2.000000]]
s=0.0 native            -> [[ 0.200000,  1.000000,   2.500000,  3.000000], [ 0.500000,  0.000000,   0.500000, -0.500000]]
s=0.0 original (off)    -> [[ 1.000000,  2.000000,   3.000000,  4.000000], [ 0.000000,  1.000000,  -1.000000,  2.000000]]
s=1.0 original          -> [[ 1.800000,  3.000000,   3.500000,  5.000000], [-0.500000,  2.000000,  -2.500000,  4.500000]]
```

The difference between the two formulations is exactly one `shift`: `original - native = eps_c - eps_w`, independent of \(s\).

With `step=5`, \(s=7.5\), the default formulation and \(\phi=0.5\), the per-sample standard deviations are
\(\operatorname{std}(\epsilon_c)=[1.290994, 1.290994]\) and, before the pull-back, \(\operatorname{std}(\text{pred})=[2.059682, 12.661252]\):

```
s=7.5 native phi=0.5    -> [[ 5.043058,  6.913870,   5.083728,  8.540663], [-1.790692,  4.132366,  -5.923058, 10.055424]]
```

Outside the window (`step=17`) or at an identity scale the output is bitwise equal to `pred_cond`, and `pred_uncond` may then be `None`.

## Constraints

- `pred_cond` must be a `torch.Tensor` with at least two dimensions (batch first) and at least 2 elements outside the batch dimension; `pred_uncond` (when guidance is on) must have the same shape. Both must be finite everywhere, otherwise `ValueError`.
- `step` and `num_inference_steps` are Python `int` values (`bool` is rejected), \(N>0\) and \(0\le\text{step}<N\).
- `start ∈ [0, 1)` (**strictly below 1**, matching the upstream constructor check), `stop ∈ [start, 1]`; `guidance_scale` is a finite real; `guidance_rescale ∈ [0, 1]`; `use_original_formulation` must be a `bool`. Any violation raises `ValueError`.
- `math.isclose` uses its default tolerances (`rel_tol=1e-9`, `abs_tol=0`), and whether it compares against \(0.0\) or \(1.0\) is decided by `use_original_formulation`.
- Intermediate math may be done in float64; return float32.
- `import diffusers` is forbidden.
