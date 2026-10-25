# Optimizer: C-Optimizer Continuous Cosine Gate with Clipping

## Problem

Cautious-style optimizers zero out coordinates where "the update direction conflicts with the gradient" via a 0/1 mask: coordinates with \(g\odot u>0\) survive, the rest are zeroed. That gate is **hard** — full trust or full cut, and the update is discontinuous when the mask flips. C-Optimizer (ByteDance Seed technical report; paper id **to be re-sourced**) replaces it with a **continuous interpolation** driven by \(\cos(g,m)\): the more the gradient and momentum align, the more passes; the more they oppose, the harder the suppression — plus a **sign-preserving element-wise clip** bounding the worst-case displacement. On the AdamW skeleton, each step is

\[
\cos=\frac{\langle g,m\rangle_F}{\|g\|_F\,\|m\|_F},\qquad
\text{gate}=\mathrm{clamp}\!\left(\frac{\cos+1}{2},0,1\right),\qquad
u=\frac{m/(1-\beta_1^t)}{\sqrt{v/(1-\beta_2^t)}+\varepsilon},\qquad
\theta\leftarrow\theta-\text{lr}\cdot\mathrm{clip}(u\cdot\text{gate},-\rho,\rho).
\]

The gate is a continuous version of the cautious mask: \(\cos=1\) fully passes (degenerating back to AdamW), \(\cos=-1\) kills everything, \(\cos=0\) passes half. Implement it: **subclass `torch.optim.Optimizer`** and update `params` in place through repeated `step()` calls on a fixed gradient sequence. CPU, float32, fully deterministic.

## Rules

- The factory takes `params` plus `lr`, `betas`, `eps`, `weight_decay`, `rho` and returns an instance of a `torch.optim.Optimizer` subclass. The exact signature, argument names and defaults live only in the starter comments and are the single source of truth.
- At step \(t\) with gradient \(g\) (state \(m\), \(v\) start at zeros; the step counter is per-parameter, starting from 1), in this **fixed order**:
  1. **decoupled weight decay** (only when `weight_decay≠0`, before any moment estimate): \(\theta\leftarrow\theta(1-\text{lr}\cdot\lambda)\);
  2. \(m\leftarrow\beta_1m+(1-\beta_1)g\); \(v\leftarrow\beta_2v+(1-\beta_2)g\odot g\);
  3. **alignment cosine** (using the **just-updated** \(m\) and the current \(g\), Frobenius inner product/norms over the whole tensor): \(\cos=\langle g,m\rangle_F/(\|g\|_F\|m\|_F)\); if either norm is 0, by convention \(\cos=1\) (no suppression);
  4. **continuous gate**: \(\text{gate}=\mathrm{clamp}((\cos+1)/2,0,1)\);
  5. **bias correction and direction** (same convention as `opt-adamw`): \(u=\dfrac{m/(1-\beta_1^t)}{\sqrt{v/(1-\beta_2^t)}+\varepsilon}\);
  6. **gate first, clip second** (fixed order): \(u\leftarrow u\cdot\text{gate}\); then \(u\leftarrow\mathrm{clip}(u,-\rho,\rho)\) — sign-preserving, element-wise;
  7. \(\theta\leftarrow\theta-\text{lr}\cdot u\).
- Without weight decay, \(|\Delta\theta|\le\text{lr}\cdot\rho\) holds every step (the gate only shrinks the displacement further).
- Parameters with `grad is None` are skipped entirely (not even the weight decay).
- Everything is float32 on CPU and deterministic; using a ready-made COPT/cautious implementation is forbidden.

## Example

Two parameters \(W_0=\begin{bmatrix}0.5&-0.2&0.1\\0.3&0.4&-0.1\end{bmatrix}\), \(b_0=[1.0,-0.5,0.2]\) with `lr=0.1`, `betas=(0.9,0.999)`, `eps=1e-8`, `weight_decay=0.01`, `\rho=0.5`, fed three gradients (top row: matrix parameter; bottom: 1-D parameter):

\(g_1: [[0.2,-0.1,0.05],[0.1,0.3,-0.2]],\ [0.1,-0.2,0.05]\); \(g_2: [[-0.15,0.2,0.1],[0.05,-0.1,0.3]],\ [-0.05,0.1,0.2]\); \(g_3: [[0.1,0.05,-0.2],[-0.3,0.15,0.1]],\ [0.2,-0.1,-0.1]\).

Step 1 in full: decay \(W\cdot0.999\); \(m_1=0.1g_1\) is strictly parallel to \(g_1\) ⇒ \(\cos=1\), \(\text{gate}=1\); after first-step bias correction \(u=g/(|g|+\varepsilon)\approx\pm1\), so **every element gets clamped by \(\rho=0.5\)** and the displacement is exactly \(\pm\text{lr}\cdot\rho=\pm0.05\). Three-step trajectory (\(W\) flattened):

```
t=1  W=[0.449500, -0.149800, 0.049900, 0.249700, 0.349600, -0.049900]   b=[0.949000, -0.449500, 0.149800]
t=2  W=[0.442595, -0.176107, -0.000150, 0.199450, 0.320328, -0.067751]   b=[0.926044, -0.427043, 0.099650]
t=3  W=[0.415041, -0.213305, 0.017276, 0.230472, 0.270242, -0.099634]   b=[0.875117, -0.388986, 0.071377]
```

Step 1 hugs the \(\pm0.05\) clip on every coordinate; from step 2 the gradients turn, \(\cos<1\), the gate starts scaling, and displacements fall below the bound — "the gate brakes on adversarial directions" is exactly this trajectory's shape.

## Constraints

- \(\rho>0\), \(0\le\beta_{1,2}<1\); `params` may mix arbitrarily shaped tensors (cos is computed per whole tensor; gate is scalar, clip element-wise)
- After every step the parameters must match an independent recomputation element by element (`atol=rtol=1e-5`)
- Three discriminating criteria the judge asserts directly: a **constant-direction** gradient sequence ⇒ \(\cos=1\), gate ≡ 1, and with \(|u|<\rho\) the whole trajectory matches native AdamW bit-for-bit (`atol=rtol=1e-6`); after 20 steps of \(+g\), one step of \(-0.5g\) ⇒ that step's displacement is \(<10^{-7}\) (the gate almost kills it); without decay, \(|\Delta\theta|\le\text{lr}\cdot\rho\) + float32 slack at every step
