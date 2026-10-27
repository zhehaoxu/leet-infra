# PEFT: VeRA Shared Frozen Base Matrices with Learnable Scaling

## Problem

**VeRA** (Kopiczko et al., ICLR 2024, arXiv:2310.11454) cuts LoRA's parameter count by another order of magnitude: all layers **share** one pair of random frozen low-rank matrices \(A\in\mathbb R^{r\times d_\text{in}}\), \(B\in\mathbb R^{d_\text{out}\times r}\), and each layer trains only two **scaling vectors** — input-side \(\lambda_d\in\mathbb R^r\) and output-side \(\lambda_b\in\mathbb R^{d_\text{out}}\):

\[\Delta W=(\text{diag}(\lambda_b)\,B)\,(\text{diag}(\lambda_d)\,A),\qquad
y=xW_0^{\!\top}+\lambda_b\odot\Bigl(\bigl(\lambda_d\odot(xA^{\!\top})\bigr)B^{\!\top}\Bigr)\]

The forward follows PEFT's `vera/layer.py`: `result + lambda_b * F.linear(lambda_d * F.linear(x, A), B)`. Each layer has only \(d_\text{out}+r\) trainable parameters, versus LoRA's \(r(d_\text{out}+d_\text{in})\).

PEFT's initialization: \(\lambda_b=0\) (the starting point is the pure base), \(\lambda_d=d_\text{initial}=0.1\).

## Task

Implement the VeRA forward and report the per-layer trainable parameter count.

- Inputs: \(W_0\in\mathbb R^{d_\text{out}\times d_\text{in}}\), \(\lambda_b\in\mathbb R^{d_\text{out}}\), \(\lambda_d\in\mathbb R^{r}\), \(A\in\mathbb R^{r\times d_\text{in}}\), \(B\in\mathbb R^{d_\text{out}\times r}\), \(x\in\mathbb R^{N\times d_\text{in}}\).
- Output \((y,\ \texttt{n\_trainable})\): \(y\in\mathbb R^{N\times d_\text{out}}\) (float32), \(\texttt{n\_trainable}=d_\text{out}+r\).

**Convention**: compute in float64 internally, output float32; broadcasting as in the formula above; no ready-made PEFT library. Tolerance \(10^{-5}\).

## Hints

- The two scalings act at different places: \(\lambda_d\) scales the low-rank **hidden** dimension (the r components), \(\lambda_b\) scales the **output** dimension (d_out components).
- With \(\lambda_b=0\) the output must equal \(xW_0^\top\) (the initialization semantics).
- You may first rebuild \(\Delta W=(\text{diag}(\lambda_b)B)(\text{diag}(\lambda_d)A)\) and verify — the functional forward and the matrix-form rebuild must agree.
