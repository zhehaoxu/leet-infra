# PEFT: RandLora Frozen Random Low-Rank Bases with Trainable Coefficients

## Description

**RandLora** (arXiv:2502.00987) makes the LoRA \(A/B\) factors **frozen random matrices** (not updated during training) and learns only a **coefficient vector** \(c\in\mathbb R^r\) of length equal to the rank \(r\) to combine these bases:

\[\Delta W = \frac{\alpha}{r}\,B\,\operatorname{diag}(c)\,A,\]

where \(A\in\mathbb R^{r\times d_\text{in}}\) and \(B\in\mathbb R^{d_\text{out}\times r}\) are deterministically generated from `base_seed` via `torch.randn`, and \(c\) is the input trainable coefficient vector. Since \(A/B\) are frozen, the actual number of trainable parameters per layer is only \(r\) scalars.

## Task

Given frozen base \(W_0\in\mathbb R^{d_\text{out}\times d_\text{in}}\), coefficient vector \(c\in\mathbb R^r\), scaling \(\alpha\), rank \(r\), and random seed `base_seed`, return the merged weight \(W'\) as follows:

1. Use `torch.Generator().manual_seed(base_seed)` to generate \(B\sim\mathcal N(0,1)\) of shape \((d_\text{out},r)\), then generate \(A\sim\mathcal N(0,1)\) of shape \((r,d_\text{in})\). **Generate B first, then A**.
2. Compute \(\Delta W = (\alpha/r)\,B\,\operatorname{diag}(c)\,A\).
3. Return \(W' = W_0 + \Delta W\).

- When \(c\) is the zero vector, \(\Delta W=0\) and \(W'=W_0\).
- When \(\alpha=0\), it also degenerates to \(W_0\).
- Output \(W'\in\mathbb R^{d_\text{out}\times d_\text{in}}\), float32.

## Hint

- The generation order of the random bases must be fixed: first generate \(B\) with shape `(d_out, r)`, then \(A\) with shape `(r, d_in)`; otherwise the same seed would produce different matrices.
- You can implement the per-rank-1-component scaling as `B @ torch.diag(c) @ A`; for larger \(r\) this is equivalent to `B * c[None, :] @ A`.
- This problem only reconstructs and merges; it does not train any parameters.
