# Optimizer: Sophia-II Clipped Second-Order Update

## Problem

Adam-family optimizers divide by the square root of the gradient's second moment \(\sqrt{v}\), so the second-order information is forever just "a moving average of gradient magnitude". Sophia (Liu et al., arXiv:**2305.14342**) replaces the denominator with the **first power of a diagonal Hessian estimate**: the preconditioner upgrades from "gradient intensity" to "curvature", and a **per-element clip** takes over the job of bounding the step when the Hessian estimate is noisy. Algorithm 3 of the paper (the Sophia-H variant) is

\[
m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t,\qquad
H_t=\beta_2 H_{t-k}+(1-\beta_2)\hat H^t,\qquad
\theta_t=\theta_{t-1}(1-\text{lr}\cdot\lambda)-\text{lr}\cdot\mathrm{clip}\!\left(\frac{m_t}{\max\{H_t,\varepsilon\}},-\rho,\rho\right),
\]

where \(\hat H\) is a Hutchinson-type random estimate — in real training it is only computed every \(k\) steps (it is expensive), and intermediate steps **freeze \(H\)** and reuse the old value. This problem makes \(\hat H\) an **explicit argument**: the judge feeds (gradient, Hessian estimate) pairs along a fixed sequence, so everything is exactly checkable on CPU.

Implement it: **subclass `torch.optim.Optimizer`** with `step(closure=None, hessians=None)` that updates `params` in place. CPU, float32, fully deterministic.

## Rules

- The factory takes `params` plus `lr`, `betas`, `eps`, `weight_decay`, `rho` and returns an instance of a `torch.optim.Optimizer` subclass. The exact signature, argument names and defaults live only in the starter comments and are the single source of truth.
- `step(closure=None, hessians=None)`:
  - `hessians` is **aligned one-to-one with the parameters** (flattened across all param groups, same order); positions of parameters without gradients still occupy a slot (the judge passes a full-length list). `hessians=None` means **no Hessian estimation this step**: \(H\) stays frozen while \(m\) and \(\theta\) update as usual — the deterministic expression of the paper's \(k\)-step reuse.
  - When `closure` is given, call it under `enable_grad` and return its value unchanged.
- Each step (for every parameter with a non-`None` `grad`), in this **fixed order**:
  1. **decoupled weight decay** (only when `weight_decay≠0`, before any moment estimate): \(\theta \leftarrow \theta\,(1-\text{lr}\cdot\lambda)\);
  2. \(m \leftarrow \beta_1 m+(1-\beta_1)g\);
  3. only when `hessians` is provided this step: \(H \leftarrow \beta_2 H+(1-\beta_2)\hat H\) (the EMA starts from all zeros);
  4. \(u \leftarrow m \,/\, \max\{H,\varepsilon\}\) (element-wise `clamp_min(ε)` on the denominator), then \(\mathrm{clip}(u,-\rho,\rho)\);
  5. \(\theta \leftarrow \theta - \text{lr}\cdot u\).
- **The denominator is the first power of \(H\)**, `max{H, ε}` — not \(\sqrt{H}\) (the square root is Adam's convention). Consequence: doubling \(\hat H\) overall **exactly halves** the unclipped displacement.
- **The clip does the bounding**: element-wise clamp to \([-\rho,\rho]\). Without weight decay, \(|\Delta\theta| \le \text{lr}\cdot\rho\) holds for every step.
- State starts at all zeros (`exp_avg`, `hessian`, `step`). \(H\)'s EMA also starts from 0: with \(\beta_2=0.99\) the first steps see \(H\) of magnitude 0.01, the denominator is mostly the \(\varepsilon\) floor, and updates are almost fully clamped by \(\rho\) — expected behavior, not a bug.
- Parameters with `grad is None` are skipped entirely (no decay, no \(m/H\) update, no write to \(\theta\)).
- Everything is float32 on CPU and deterministic; using a ready-made Sophia implementation is forbidden.

## Example

Single parameter \(\theta_0=[1.0,\,-2.0,\,0.5]\) with `lr=0.1`, `betas=(0.9,0.99)`, `eps=1e-8`, `weight_decay=0.1`, `\rho=0.5`, fed three (gradient, \(\hat H\)) pairs:

\(g_1=[0.10,-0.20,0.30]\), \(\hat H_1=[0.5,1.0,0.25]\); \(g_2=[-0.40,0.05,0.20]\), \(\hat H_2=[0.5,0.5,0.5]\); \(g_3=[0.15,-0.10,-0.25]\), \(\hat H_3=[2.0,1.0,0.1]\).

Step 1 in full: weight decay \(\theta\cdot0.99=[0.99,-1.98,0.495]\); \(m_1=0.1g_1=[0.01,-0.02,0.03]\); \(H_1=0.01\hat H_1=[0.005,0.01,0.0025]\); \(m/\max\{H,\varepsilon\}=[2,-2,12]\) — **all three components get clamped by \(\rho=0.5\)**, landing at \(\theta_1=[0.94,\,-1.93,\,0.445]\). The three-step trajectory:

```
t=1  θ=[0.940000, -1.930000, 0.445000]
t=2  θ=[0.980600, -1.860700, 0.390550]
t=3  θ=[1.014009, -1.792093, 0.336644]
```

Every direction of step 1 is clamped (the EMA starts from zero ⇒ the early denominator is tiny) — a live specimen of "the first few steps are bounded by \(\rho\)". By step 3 \(H_3=[0.0247,0.0099,0.0026]\) is still small, so the displacement remains clip-bound.

## Constraints

- \(\rho>0\), \(0\le\beta_{1,2}<1\); `params` may mix arbitrarily shaped tensors (denominator and clip are element-wise)
- After every step the parameters must match an independent recomputation element by element (`atol=rtol=1e-5`; degenerations below)
- Three discriminating criteria the judge asserts directly: doubling \(\hat H\) ⇒ the displacement is exactly halved (a \(\sqrt{H}\) denominator gives \(1/\sqrt2\) instead); with \(H\approx0\) the displacement is exactly \(\text{lr}\cdot\rho\) in the direction opposite the sign of \(m\), and \(|\Delta\theta|\le\text{lr}\cdot\rho\) holds at every step; with \(\beta_1=\beta_2=0\), \(\hat H\equiv1\), huge \(\rho\) and no decay the update degenerates bit-for-bit into native SGD (you must take \(\beta_2=0\): the EMA starts from 0, so with \(\beta_2=0.99\) the first few \(H\) values are of magnitude 0.01 and the degeneration fails)
- On a `hessians=None` step: \(H\) is bit-identical afterwards, while \(m\) and \(\theta\) update as usual
