# LLM Component: RMSNorm

## Problem Description

**RMSNorm (Root Mean Square Layer Normalization)** is the standard normalization layer in modern LLMs such as LLaMA / Qwen / Mistral / DeepSeek. Compared with LayerNorm, it **does not subtract the mean and does not estimate a bias**, scaling only by the root mean square (RMS) — cheaper to compute and more stable.

$$ \text{out}_i = w_i \cdot \frac{x_i}{\sqrt{\dfrac{1}{d}\sum_{j=1}^{d} x_j^2 + \epsilon}} $$

> This collection implements the LLM component by component, following `transformers`' own **component** decomposition, and finally assembles the full model. The judge environment has `transformers` installed,
> and on submission your implementation is compared element-wise (`allclose`) against **`transformers`' `Qwen2RMSNorm` (same weights)** —
> so what must be reproduced is its **true semantics**, not just the formula.

## Rules

The inputs are the tensor to normalize $x$ (shape $(\dots, d)$, the last dimension being the feature dimension), the element-wise learnable weight $w$ (shape $(d,)$, i.e. `weight` in HF), and a float $\epsilon$; return the normalized result with the **same shape** as $x$. Key semantic points:

- The variance is the **mean square over the last dimension** $\frac{1}{d}\sum_j x_j^2$, with **no mean subtraction** (this is the fundamental difference from LayerNorm).
- $\epsilon$ is **added inside the square root**: $\sqrt{\text{ms} + \epsilon}$.
- HF computes the mean square and the normalization in **float32**, casts back to the original dtype, and only then multiplies by `weight` — follow this to guarantee numerical consistency.
- Do not modify the input tensor.

## Example

```
x = [1, 2, 3, 4], weight = [1, 1, 1, 1], eps = 1e-6
mean square = (1+4+9+16)/4 = 7.5, RMS = sqrt(7.5 + 1e-6) ≈ 2.7386
output ≈ [0.365148, 0.730297, 1.095445, 1.460593]
```

## Constraints

- $1 \le d \le 4096$; $x$ has at least 1 dimension, with the last one being $d$; `weight` has shape $(d,)$
- dtype is mainly float32; keep the output consistent with the input
- Cross-checked with `torch.allclose` (relative tolerance $10^{-5}$) against the `transformers` reference implementation
- Do not directly call `transformers` modules as a "shortcut" — that is the judge's oracle
