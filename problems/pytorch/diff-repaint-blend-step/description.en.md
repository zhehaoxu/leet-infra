# Diffusion Sampling: RePaint's Masked Blend Step Reads the Previous Alpha, Not the Current One

## Problem

When RePaint inpaints an image, every step does two opposite things at once: it denoises the current sample toward \(x_{t-\text{prev}}\) exactly like DDPM/DDIM would, but that formula is only allowed to act on **the region that needs to be filled in**; the known region marked by the mask never enters the network at all — the original image is simply re-noised to the same time point, and the two halves are stitched together cell by cell. Get the blend direction wrong, or let the network output leak into the known region, and the inpainted image stops matching the original.

diffusers' `RePaintScheduler` also ships the reverse operation, `undo_step`: push the sample **forward** by \(n = T/N\) sub-steps, re-noising one \(\beta\) at a time, used when the sampler has to jump back up to a time point it already left. It uses the forward diffusion's own coefficients — one \(\beta\) per sub-step — not a single big jump.

This problem joins the two halves, and turns everything the upstream scheduler generates on its own into explicit inputs: the \(\bar\alpha\) table comes in as a 1-D `betas`, and both noise draws (one for the step, \(n\) for the undo) are given too. There is no real network and no random number generation.

## Rules

Write \(T=\)`len(betas)`, \(n=T\,//\,N\) (integer division), \(\bar\alpha_t=\prod_{j\le t}(1-\beta_j)\) (prefix product, index \(t\) inclusive); `timestep` indexes the \(\beta\) and \(\bar\alpha\) tables directly. Below, \(x/\epsilon/x_{\text{orig}}/z\) stand for `sample / model_output / original_image / noise`. All intermediates are computed in float64.

**Denoise and blend** (\(t\to t-n\)):

\[ \text{prev}=t-n,\qquad \bar\alpha_{\text{prev}}=\begin{cases}\bar\alpha_{t-n} & \text{prev}\ge 0\\[2pt] 1.0 & \text{prev}<0\end{cases} \]

\[ \hat{x}_0=\frac{x-\sqrt{1-\bar\alpha_t}\;\epsilon}{\sqrt{\bar\alpha_t}},\qquad \text{clamped to } [-1,1] \text{ when clip\_sample} \]

\[ \tilde\beta=\frac{1-\bar\alpha_{\text{prev}}}{1-\bar\alpha_t}\Big(1-\frac{\bar\alpha_t}{\bar\alpha_{\text{prev}}}\Big),\qquad \sigma=\eta\sqrt{\tilde\beta},\qquad \text{add}=\begin{cases}\sigma\,z & t>0 \text{ and } \eta>0\\ 0 & \text{otherwise}\end{cases} \]

\[ \text{unknown}=\sqrt{\bar\alpha_{\text{prev}}}\,\hat{x}_0+\sqrt{1-\bar\alpha_{\text{prev}}-\sigma^2}\;\epsilon+\text{add} \]

\[ \text{known}=\sqrt{\bar\alpha_{\text{prev}}}\,x_{\text{orig}}+\sqrt{1-\bar\alpha_{\text{prev}}}\;z \]

\[ \text{prev\_sample}=\text{mask}\cdot\text{known}+(1-\text{mask})\cdot\text{unknown} \]

**Forward undo** (starting from `sample`, walking \(n\) steps one \(\beta\) at a time):

\[ x \leftarrow \sqrt{1-\beta_{t+i}}\;x+\sqrt{\beta_{t+i}}\;\text{undo\_noise}_i,\qquad i=0,\dots,n-1 \]

Five places that are easy to get wrong:

1. In `mask`, **1 means "keep the original"** and 0 is the region to fill; the known branch uses \(\bar\alpha_{\text{prev}}\), not \(\bar\alpha_t\).
2. \(\sigma=\eta\sqrt{\tilde\beta}\) (dropping the square root turns \(\eta{=}0.5\) into \(\eta\tilde\beta\)), and the square root in the direction term subtracts \(\sigma^2\) from \(1-\bar\alpha_{\text{prev}}\) — again \(\bar\alpha_{\text{prev}}\), not \(\bar\alpha_t\).
3. The \(\eta\) term enters the **unknown** branch only when \(t>0\) and \(\eta>0\); the known branch's \(\sqrt{1-\bar\alpha_{\text{prev}}}\,z\) is added at every \(t\). Both branches use the **same** `noise` tensor.
4. When \(\text{prev}<0\), \(\bar\alpha_{\text{prev}}\) is **1.0** (the `final_alpha_cumprod` hardcoded upstream), not \(\bar\alpha_0\). Then `unknown` collapses to \(\hat x_0\), `known` collapses to the original image, and the whole expression becomes `mask*orig + (1-mask)*x0_hat`.
5. The undo's accumulated coefficient is \(\sqrt{\prod_{i=0}^{n-1}(1-\beta_{t+i})}=\sqrt{\bar\alpha_{t+n-1}/\bar\alpha_{t-1}}\) (with \(\bar\alpha_{-1}:=1\) when \(t=0\)); writing \(\sqrt{\bar\alpha_{t+n}/\bar\alpha_t}\) slips in one extra \(\sqrt{\alpha_{t+n}}\).

Return the triple `(prev_sample, x0_hat, undone)`, all float32 tensors with the same shape as `sample`; the second entry is \(\hat x_0\) (possibly already clamped).

## Example

```
betas    = [0.05, 0.08, 0.12, 0.18, 0.25, 0.33, 0.40, 0.50]   # T = 8
abar     = [0.95, 0.874, 0.76912, 0.630678, 0.473009, 0.316916, 0.19015, 0.095075]
N = 4  ->  n = 2
```

With `timestep=4`, `eta=0.5`, `clip_sample=True`, the intermediates are:

```
abar_t(4)      = 0.473009   sqrt = 0.687756    1 - abar_t     = 0.526991   sqrt = 0.725942
abar_prev(2)   = 0.769120   sqrt = 0.876995    1 - abar_prev  = 0.230880   sqrt = 0.480500
beta_tilde     = 0.168672   sqrt = 0.410697    sigma = 0.205349    sigma^2 = 0.042168
direction_coef = sqrt(0.230880 - 0.042168) = sqrt(0.188712) = 0.434410
x0_hat_preclip = [[0.053841, -0.753922], [1.534764, 0.185249], [0.197112, 0.886394]]
undo_coef      = sqrt((1-0.25)*(1-0.33)) = sqrt(0.502500) = 0.708872
```

Given

```
sample         = [[0.4, -0.7], [1.6, 0.2], [-0.3, 0.9]]
model_output   = [[0.5, -0.25], [0.75, 0.1], [-0.6, 0.4]]
original_image = [[-0.8, 0.6], [0.3, -0.5], [0.9, 0.2]]
mask           = [[1, 0], [0, 1], [1, 0]]
noise          = [[0.7, -0.4], [0.2, 0.9], [-0.5, 0.3]]
undo_noise     = [[[0.1, 0.2], [-0.3, 0.4], [0.5, -0.1]],
                  [[-0.2, 0.3], [0.4, -0.5], [0.1, 0.6]]]
```

then:

```
prev_sample          = [[-0.365246, -0.851928], [1.243872, -0.006048], [0.549046, 1.012732]]
x0_hat               = [[0.053841, -0.753922], [1.0, 0.185249], [0.197112, 0.886394]]
undone               = [[0.209584, -0.24202], [1.241198, 0.018253], [0.049418, 0.941732]]
```

`prev_sample[0][0]` sits under the mask, so it is fully determined by the original image and the noise: \(0.876995\times(-0.8)+0.480500\times 0.7=-0.365246\), and `model_output` never touches it.

The three masked cells (`mask=1`) stay bit-for-bit identical under any combination of `eta`/`clip_sample` — the known branch takes neither \(\eta\) nor the network — while every unmasked cell moves with \(\eta\): the coefficient on \(\epsilon\) goes from \(0.434410\) to \(\sqrt{0.230880}=0.480500\) and the \(\sigma z\) term disappears. Here the ceiling for \(\eta\) is \(\sqrt{(1-\bar\alpha_t)/(1-\bar\alpha_t/\bar\alpha_{\text{prev}})}=1.169961\); beyond it the call errors out.

```
clip_sample=False:  x0_hat[1][0] = 1.534764    prev_sample[1][0] = 1.712857
eta=0.0:            prev_sample = [[-0.365246, -0.781311], [1.23737, -0.006048], [0.549046, 0.969563]]
eta=1.0:            prev_sample = [[-0.365246, -0.887818], [1.146195, -0.006048], [0.549046, 1.000338]]
eta=1.5:            ValueError
```

When `timestep` falls before the first jump (\(t<n\), here \(t\in\{0,1\}\)) we get \(\bar\alpha_{\text{prev}}=1\), `eta` stops mattering entirely, and

```
t=1: prev_sample = [[-0.8, -0.653837], [1.0, -0.5], [0.9, 0.810815]]
```

that row is bit-for-bit `mask * original_image + (1 - mask) * x0_hat` — `x0_hat` still uses the table values at \(t=1\), and the `[1][0]` cell is clamped to \(1.0\).

## Constraints

- `sample / model_output / original_image / mask / noise` must be tensors of one and the same shape (dimension 0 is the batch, at least 1 dimension) and all finite; `mask` values must additionally lie in \([0,1]\).
- `betas` is a finite 1-D tensor with elements in \([0,1)\), and its prefix product \(\bar\alpha\) may not reach 0.
- `undo_noise` must have shape `(n,) + sample.shape`.
- `timestep` and `num_inference_steps` must be Python ints (`bool` is illegal); \(N\ge 1\) and \(T\bmod N=0\); \(0\le t\) and \(t+n\le T\) (otherwise `undo_step` indexes past the table).
- `eta` is a finite non-negative real (it may exceed 1) but must satisfy \(1-\bar\alpha_{\text{prev}}-\sigma^2\ge 0\); `clip_sample` is a bool.
- Intermediates in float64, all three return values float32. `import diffusers` is forbidden.
