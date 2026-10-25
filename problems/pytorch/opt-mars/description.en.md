# Optimizer: MARS Variance-Reduced Momentum with Trust Ratio

## Problem

The variance of stochastic gradient estimates is a hidden tax on large-model training: momentum averages some of it away, but still overreacts on steps where the gradient suddenly turns. MARS (*Make vAriance Reduction Shine*, Yuan et al., arXiv:**2411.10438**) ports the **variance-reduction (STB) correction** from stochastic optimization into Adam-style updates: build a **variance-reduced gradient estimate** \(c_t\) from the difference of consecutive gradients \((g_t-g_{t-1})\), then apply a **per-tensor trust-ratio clip** \(\rho_t=\min(1,1/\|c_t\|_F)\), and let the clipped \(c̃_t\) drive **both** the first and second moments. Instantiated as MARS-AdamW:

\[
c_t=g_t+\gamma'\,(g_t-g_{t-1}),\qquad \gamma'=\frac{\gamma\beta_1}{1-\beta_1},\qquad
c̃_t=\frac{c_t}{\max(1,\|c_t\|_F)},\qquad
\theta_t\leftarrow\theta_{t-1}-\frac{\text{lr}}{1-\beta_1^t}\cdot\frac{m_t}{\sqrt{v_t/(1-\beta_2^t)}+\varepsilon}
\]

with \(m_t=\beta_1m_{t-1}+(1-\beta_1)c̃_t\) and \(v_t=\beta_2v_{t-1}+(1-\beta_2)c̃_t^{\odot2}\) — the second moment's driving signal is also the clipped \(c̃\), not the raw \(g\). Implement it: **subclass `torch.optim.Optimizer`** and update `params` in place through repeated `step()` calls on a fixed gradient sequence. CPU, float32, fully deterministic.

## Rules

- The factory takes `params` plus `lr`, `betas`, `gamma`, `eps`, `weight_decay`, `ams_bound` and returns an instance of a `torch.optim.Optimizer` subclass. The exact signature, argument names and defaults live only in the starter comments and are the single source of truth.
- State starts at all zeros: \(m\), \(v\), `last_g` (plus \(v_{\max}\) when `ams_bound=True`). `last_g` starts at 0 — the **first step assumes \(g_0:=0\)**, so the first difference term \(g_1-g_0=g_1\) is nonzero.
- At step \(t\) with gradient \(g\) (per-parameter counter starting from 1), in this **fixed order**:
  1. **decoupled weight decay** (only when `weight_decay≠0`, before any moment estimate): \(\theta\leftarrow\theta(1-\text{lr}\cdot\lambda)\);
  2. **variance-reduced estimate**: \(c=g+\gamma'(g-\text{last\_g})\) with \(\gamma'=\gamma\beta_1/(1-\beta_1)\) (first step: \(\text{last\_g}=0\Rightarrow c=(1+\gamma')g\));
  3. **trust-ratio clip** (per tensor): \(\rho=\min(1,\ 1/\|c\|_F)\), \(c̃=\rho\cdot c\) (equivalent to \(c/\max(1,\|c\|_F)\); \(\|\cdot\|_F\) is the whole-tensor 2-norm);
  4. \(m\leftarrow\beta_1m+(1-\beta_1)c̃\); \(v\leftarrow\beta_2v+(1-\beta_2)c̃\odot c̃\);
  5. **adaptive clipping of the second moment** (only when `ams_bound=True`): \(v_{\max}\leftarrow\max(v_{\max},v)\), denominator base \(v_{\max}\); otherwise \(v\);
  6. **bias correction and update** (same convention as `opt-adamw`): \(\theta\leftarrow\theta-\dfrac{\text{lr}}{1-\beta_1^t}\cdot\dfrac{m}{\sqrt{\text{denom\_base}/(1-\beta_2^t)}+\varepsilon}\);
  7. `last_g ← g`.
- Parameters with `grad is None` are skipped entirely (no decay, no counter advance, no `last_g` update).
- 1-D and multi-dimensional parameters follow **exactly the same path** (the trust ratio is computed per whole tensor, no shape branch).
- Everything is float32 on CPU and deterministic; using a ready-made MARS implementation is forbidden.

## Example

Two parameters \(W_0=\begin{bmatrix}0.5&-0.2&0.1\\0.3&0.4&-0.1\end{bmatrix}\), \(b_0=[1.0,-0.5,0.2]\) with `lr=0.1`, `betas=(0.9,0.99)`, `gamma=0.025`, `eps=1e-8`, `weight_decay=0.01`, `ams_bound=False`, fed the same three gradients \(g_1,g_2,g_3\) as the opt-copt example. \(\gamma'=0.025\cdot0.9/0.1=0.225\).

Step 1 in full: decay \(W\cdot0.999\); \(c=(1+0.225)g_1\), and both tensors' \(\|c\|_F\approx0.538,0.281\) are below 1 ⇒ the trust ratio never triggers; after first-step bias correction the update is \(\approx0.1\cdot\mathrm{sign}(g_1)\). Three-step trajectory (\(W\) flattened):

```
t=1  W=[0.399500, -0.099800, -0.000100, 0.199700, 0.299600, 0.000100]   b=[0.899000, -0.399500, 0.099800]
t=2  W=[0.400933, -0.139417, -0.097481, 0.113004, 0.273941, -0.029651]   b=[0.884797, -0.385796, 0.010913]
t=3  W=[0.374778, -0.173529, -0.072064, 0.151830, 0.227537, -0.059808]   b=[0.823631, -0.348057, -0.009175]
```

From step 2 the difference term \(g_t-g_{t-1}\) enters \(c\) and the momentum starts carrying "gradient rate of change" — the trajectory folds back and forth inside the trust band.

## Constraints

- \(\gamma\ge0\), \(0\le\beta_{1,2}<1\); `params` may mix 1-D and multi-dimensional tensors
- After every step the parameters must match an independent recomputation element by element (`atol=rtol=1e-5`)
- Three discriminating criteria the judge asserts directly: `gamma=0` ⇒ \(c\equiv g\), and with gradient norms below 1 throughout the trust ratio never triggers ⇒ bit-for-bit match with **native AdamW** (`atol=rtol=1e-7`); a first step with \(\|c\|_F>1\) ⇒ \(\|c̃\|_F=1\) and \(m_1\)'s norm is exactly \((1-\beta_1)\) (independent of the gradient scale); with `ams_bound=True` the denominator is never below the historical peak (adaptive clipping of the second moment)
