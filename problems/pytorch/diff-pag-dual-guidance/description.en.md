# Diffusion: PAG Dual-Scale Guidance and Its Two Windows

## Problem Statement

**Perturbed Attention Guidance** (PAG) runs one extra pass through a deliberately degraded model: a few self-attention layers are replaced by a near-identity map, which yields a lower-quality conditional prediction \(\epsilon_{\text{skip}}\) that still keeps the coarse structure. The guidance direction is then built the same way CFG builds it — "good prediction minus bad prediction" — except the reference branch is the perturbed one instead of the unconditional one:

\[ \Delta_{\text{cfg}}=\epsilon_c-\epsilon_u,\qquad \Delta_{\text{pag}}=\epsilon_c-\epsilon_{\text{skip}} \]

The two PAG scales also have independent active ranges: CFG is normally on for the whole trajectory, while PAG is only on during the early part of denoising (by default the \(1\%\sim 20\%\) step range) — that is when the structure is not settled yet and the perturbed difference is most informative. In diffusers the two ranges do **not** share endpoint semantics, and getting this branch/window logic exactly right is the point of this problem.

## Rules

Given three same-shaped predictions (dim 0 is the batch):

- `pred_cond` \(\epsilon_c\): the conditional prediction;
- `pred_uncond` \(\epsilon_u\): the unconditional prediction;
- `pred_cond_skip` \(\epsilon_{\text{skip}}\): the conditional prediction of the perturbed-attention branch.

together with the scales `guidance_scale = s`, `perturbed_scale = s_{\text{skip}}`, the current `step`, the total `num_inference_steps = N`, and the two interval parameter pairs `start / stop` and `perturbed_start / perturbed_stop`.

First decide whether each branch is active:

\[ \text{cfg\_on} = \big[\lfloor \text{start}\cdot N\rfloor \le \text{step} < \lfloor \text{stop}\cdot N\rfloor\big] \ \wedge\ \lnot\text{isclose}(s, 1) \]

\[ \text{pag\_on} = \big[\lfloor \text{perturbed\_start}\cdot N\rfloor < \text{step} < \lfloor \text{perturbed\_stop}\cdot N\rfloor\big] \ \wedge\ \lnot\text{isclose}(s_{\text{skip}}, 0) \]

The CFG interval is **left-closed, right-open**, the PAG interval is **open on both ends**. `isclose` means `math.isclose` with its default tolerances (`1e-9` relative, `0` absolute), i.e. CFG counts as off when `s` is close to `1.0` and PAG counts as off when `s_skip` is close to `0.0`.

Then pick the branch:

| cfg_on | pag_on | output |
| --- | --- | --- |
| no | no | \(\epsilon_c\) |
| no | yes | \(\epsilon_{\text{skip}} + s_{\text{skip}}\,\Delta_{\text{pag}}\) |
| yes | no | \(\epsilon_u + s\,\Delta_{\text{cfg}}\) |
| yes | yes | \(\epsilon_u + s\,\Delta_{\text{cfg}} + s_{\text{skip}}\,\Delta_{\text{pag}}\) |

Return a float32 tensor of the same shape as `pred_cond`. When a branch is off, its input may be passed as `None`; if a branch is active but its input is `None`, raise `ValueError`.

## Example

\[ \epsilon_c=\begin{bmatrix}0.4&-1.2&0.6\\0.0&0.5&-0.5\end{bmatrix},\ \epsilon_u=\begin{bmatrix}0.1&-0.4&0.2\\0.2&-0.1&0.0\end{bmatrix},\ \epsilon_{\text{skip}}=\begin{bmatrix}0.2&-0.8&0.5\\0.1&0.3&-0.2\end{bmatrix} \]

With \(s=7.5\), \(s_{\text{skip}}=2.8\), \(N=10\) and the default intervals (CFG \([0,10)\), PAG \((0,2)\)):

```
step=1 -> [[ 2.910000, -7.520000,  3.480000], [-1.580000,  4.960000, -4.590000]]
step=0 -> [[ 2.350000, -6.400000,  3.200000], [-1.300000,  4.400000, -3.750000]]
step=5 -> [[ 2.350000, -6.400000,  3.200000], [-1.300000,  4.400000, -3.750000]]
```

`step=1` has both branches active; `step=0` falls outside the open PAG interval, so only CFG remains and the result is bit-identical to `step=5`.

## Constraints

- All three inputs must have at least one dimension (dim 0 is the batch) and identical shapes; a non-tensor input or a shape mismatch raises `ValueError`.
- `guidance_scale` and `perturbed_scale` are finite non-negative numbers; `step` and `num_inference_steps` are integers (`bool` does not count), with \(N>0\) and \(0\le \text{step} < N\).
- `start`, `stop`, `perturbed_start`, `perturbed_stop` all lie in \([0,1]\) with `start <= stop` for each pair; otherwise raise `ValueError`.
- Windows are computed by truncating `int(...)` (not rounding), so for very few steps the PAG interval can be empty.
- **Degenerate cases**: \(s=1\) turns CFG off; \(s_{\text{skip}}=0\) turns PAG off; when both are off the output must equal `pred_cond` bit for bit.
- Compute in float64 internally if you like, but return float32.
