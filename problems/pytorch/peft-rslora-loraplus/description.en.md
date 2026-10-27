# PEFT: rsLoRA Scaling and LoRA+ Divergent Learning Rates

## Problem

The LoRA update \(\Delta W=\frac{\alpha}{r}BA\) hides two "hyperparameter traps" that were each fixed by later work:

- **rsLoRA** (Kalajdzievski, arXiv:2312.03732) points out that \(\alpha/r\) makes the update's effective norm decay like \(1/r\) at **high rank**, faster than the \(1/\sqrt d\) scale inherent to weight matrices — LoRA's stability degrades as the rank grows. Changing the scale to \(\alpha/\sqrt r\) makes high-rank training more stable and converges faster.
- **LoRA+** (Hayou et al., arXiv:2402.12354) points out that when \(A\) (down-projection) and \(B\) (up-projection) share one learning rate, the **feature learning rate** of finite-width networks is asymmetric: the \(B\) side needs a larger learning rate \(\eta_B=\lambda\,\eta_A\) (the paper recommends \(\lambda=16\)), otherwise low-rank fine-tuning underfits.

The two papers differ in mechanism (inference-time scaling vs training-time optimizer), but both can be observed on "the update after one SGD step":

\[A'=A-\eta_A\,g_A,\qquad B'=B-\lambda\,\eta_A\,g_B,\qquad \Delta W=B'A'\]

## Task

Given the LoRA factors and one gradient step, perform the SGD update and output the update amount under the rsLoRA scale and the vanilla scale respectively.

- Inputs \(A\in\mathbb R^{r\times d_\text{in}}\), \(B\in\mathbb R^{d_\text{out}\times r}\), gradients \(g_A,g_B\) of the same shapes; \(\eta_A=\texttt{lr}\), \(\lambda=\texttt{lam}\), \(\alpha=\texttt{alpha}\) are real numbers and \(r\) a positive integer.
- Output \((\Delta_\text{rs},\ \Delta_\text{vanilla})\), both \(\mathbb R^{d_\text{out}\times d_\text{in}}\):

\[\Delta_\text{rs}=\frac{\alpha}{\sqrt r}\,B'A',\qquad \Delta_\text{vanilla}=\frac{\alpha}{r}\,B'A'\]

**Convention**: compute in float64 internally, output float32; \(\sqrt r\) as a float64 square root; do not use any ready-made PEFT library. Tolerance \(10^{-5}\).

## Hints

- The \(A\) side uses \(\eta_A\), the \(B\) side uses \(\lambda\eta_A\) — \(\lambda\) only enters the update of \(B\).
- The two outputs share the same \(B'A'\) and differ by a scalar factor of \(\sqrt r\).
- With \(g_B=0\), changing \(\lambda\) must not change the output at all (the asymmetry of LoRA+).
