# Quantization: BitNet b1.58 Ternary Quantization and Add-Subtract Count GEMM

## Description

**BitNet b1.58** (Ma et al., arXiv:2402.17764) constrains weights to \(\{-1,0,1\}\) and uses the **absmean** of a block as the scale factor, turning matrix multiplication into a lookup-table-style add/subtract count and greatly reducing multiply overhead during large-model inference.

Given a weight matrix \(W\in\mathbb R^{d_\text{out}\times d_\text{in}}\):

1. Compute the scale
   \[s=\frac{1}{d_\text{out}\cdot d_\text{in}}\sum_{i,j}|W_{ij}|.\]
2. Ternarize
   \[T_{ij}=\operatorname{clamp}\left(\operatorname{round}\left(\frac{W_{ij}}{s}\right), -1, 1\right),\quad T_{ij}\in\{-1,0,1\}.\]
3. Dequantize
   \[\hat W_{ij}=s\cdot T_{ij}.\]
4. Run ternary GEMM for input \(X\in\mathbb R^{b\times d_\text{in}}\):
   \[Y=X\,\hat W^{\mathsf T}=s\left(\sum_{j:T_{ij}=1}X_{:,j}-\sum_{j:T_{ij}=-1}X_{:,j}\right)\in\mathbb R^{b\times d_\text{out}}.\]

## Task

Implement `solve(W, X)`:

- Inputs: `W` (float32 \((d_\text{out}, d_\text{in})\)), `X` (float32 \((b, d_\text{in})\)).
- Return `(s, T, W_hat, Y)`:
  - `s` is a float32 scalar;
  - `T` is int64 with shape \((d_\text{out}, d_\text{in})\);
  - `W_hat` is float32 with the same shape as `W`;
  - `Y` is float32 with shape \((b, d_\text{out})\).

## Hints

- Clamp the rounded values to \(\{-1,0,1\}\) to obtain the ternary tensor; zeros stay zero after rounding.
- For the GEMM, sum `X` over the positive and negative masks separately and subtract, avoiding a large dense intermediate tensor.
- Tolerance \(10^{-5}\).
