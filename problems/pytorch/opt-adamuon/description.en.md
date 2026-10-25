# Optimizer: AdaMuon Element-wise Adaptive Orthogonal Update

## Problem

Muon brought the "gradient → momentum → Newton-Schulz orthogonalization" orthogonal update into large-model training: the update direction is flattened across the matrix spectrum so every singular direction moves equally. But it treats all coordinates alike — none of Adam's per-coordinate adaptive step size. AdaMuon (Si et al., arXiv:**2507.11005**) stacks an **element-wise RMS preconditioner** on top of Muon, then performs a **norm restoration** that preserves Muon's overall update scale exactly:

\[
m\leftarrow\mu m+(1-\mu)g,\quad G=(1-\mu)g+\mu m,\quad O=\mathrm{NS}(G),\quad
v\leftarrow\beta_2v+(1-\beta_2)O^{\odot2},\quad
\theta\leftarrow\theta-\text{lr}\cdot U,
\]
\[
U=\frac{O}{\sqrt{v/(1-\beta_2^t)}+\varepsilon}\cdot\frac{\|O\|_F}{\|O/(\sqrt{v/(1-\beta_2^t)}+\varepsilon)\|_F}.
\]

Two key designs: **the second moment's driving signal is the orthogonalized \(O\) (not \(g\))** — adaptivity happens after Muon; and **the RMS division is immediately followed by norm restoration**, so \(\|\Delta\theta\|_F=\text{lr}\cdot\|O\|_F\) always holds — Muon's spectral scale is not destroyed by the per-coordinate division. 1-D parameters (biases, norm gains) take a pure AdamW path. Implement it: **subclass `torch.optim.Optimizer`** and update `params` in place through repeated `step()` calls on a fixed gradient sequence. CPU, float32, fully deterministic.

## Rules

- The factory takes `params` plus `lr`, `momentum`, `ns_steps`, `betas`, `eps`, `weight_decay` and returns an instance of a `torch.optim.Optimizer` subclass. The exact signature, argument names and defaults live only in the starter comments and are the single source of truth.
- State starts at all zeros (\(m\), \(v\)); the step counter is per-parameter, starting from 1.
- **Decoupled weight decay** (only when `weight_decay≠0`, before any update): \(\theta\leftarrow\theta(1-\text{lr}\cdot\lambda)\).
- **2-D parameters (dim ≥ 2)**, in this **fixed order**:
  1. momentum: \(m\leftarrow\mu m+(1-\mu)g\);
  2. Nesterov: \(G\leftarrow(1-\mu)g+\mu m\) (using the updated \(m\));
  3. Newton-Schulz orthogonalization \(O=\mathrm{NS}(G)\) with quintic coefficients \((3.4445,-4.7750,2.0315)\), `ns_steps` iterations: transpose first when rows>cols; \(X\leftarrow X/(\|X\|_F+10^{-7})\); loop \(\{A=XX^{\!\top};\ X=aX+(bA+cA^2)X\}\); transpose back if transposed. **No** Muon \(\max(1,\text{rows}/\text{cols})^{0.5}\) spectral scaling (norm restoration cancels it; stacking both is double-scaling);
  4. element-wise RMS second moment: \(v\leftarrow\beta_2v+(1-\beta_2)O\odot O\), \(\hat V=v/(1-\beta_2^t)\);
  5. **adaptive preconditioning + norm restoration** (fixed order): \(U\leftarrow O/(\sqrt{\hat V}+\varepsilon)\); then \(U\leftarrow U\cdot\big(\|O\|_F/(\|U\|_F+10^{-12})\big)\);
  6. \(\theta\leftarrow\theta-\text{lr}\cdot U\).
- **1-D parameters (dim < 2) take a pure AdamW path**: \(m\leftarrow\beta_1m+(1-\beta_1)g\), \(v\leftarrow\beta_2v+(1-\beta_2)g^{\odot2}\), \(\theta\leftarrow\theta-\text{lr}\cdot\dfrac{m/(1-\beta_1^t)}{\sqrt{v/(1-\beta_2^t)}+\varepsilon}\) (`momentum` is only the 2-D momentum coefficient).
- Parameters with `grad is None` are skipped entirely. Without weight decay, \(\|\Delta\theta\|_F=\text{lr}\cdot\|O\|_F\) holds at every step.
- Everything is float32 on CPU and deterministic; ready-made Muon/AdaMuon/AdamW combinators are forbidden (implement NS as a hand-written loop).

## Example

Two parameters \(W_0=\begin{bmatrix}0.5&-0.2&0.1\\0.3&0.4&-0.1\end{bmatrix}\) (2-D, Muon+Ada path) and \(b_0=[1.0,-0.5,0.2]\) (1-D, AdamW path) with `lr=0.1`, `momentum=0.95`, `ns_steps=5`, `betas=(0.9,0.95)`, `eps=1e-8`, `weight_decay=0.01`, fed the same three gradients as the opt-mars example.

Step 1: on the \(b\) side AdamW's first step gives \(u\approx\mathrm{sign}(g)\), displacement \(\pm0.1\) (\(b_1=[0.899,-0.3995,0.0998]\)). On the \(W\) side the first step has \(\hat V=O^{\odot2}\) ⇒ \(U\approx\mathrm{sign}(O)\) with equal magnitudes element-wise, then norm restoration — **the first-step displacements are all equal**:

```
t=1  W=[0.451636, -0.151936, 0.052036, 0.251836, 0.351736, -0.052036]   b=[0.899000, -0.399500, 0.099800]
t=2  W=[0.489015, -0.213749, -0.006608, 0.193297, 0.364099, -0.106108]   b=[0.871264, -0.372264, 0.012227]
t=3  W=[0.447945, -0.274676, 0.071185, 0.238998, 0.308142, -0.126250]   b=[0.805518, -0.326431, -0.021284]
```

All six of \(W\)'s first-step displacements are \(\pm0.047864\) (a sign plate) — the direct consequence of "first step \(\sqrt{\hat V}=|O|\)". From step 2 \(v\) accumulates per-coordinate differences and the displacements stop being equal, yet the whole step's \(\|\Delta W\|_F\) stays pinned at \(\text{lr}\cdot\|O\|_F\) by the restoration.

## Constraints

- \(\mu\in[0,1)\), \(0\le\beta_{1,2}<1\), `ns_steps≥1`; `params` may mix 1-D and ≥2-D tensors
- After every step the parameters must match an independent recomputation element by element (`atol=rtol=1e-5`)
- Three discriminating criteria the judge asserts directly: 1-D parameters match **native `torch.optim.AdamW`** bit-for-bit (`atol=rtol=1e-6`); the first 2-D step's displacements are element-wise equal (sign plate, with \(\varepsilon=10^{-12}\) within slack); without decay \(\|\Delta\theta\|_F=\text{lr}\cdot\|O\|_F\) at every step (\(O\) recomputed with an **independently implemented NS** inside the tests)
