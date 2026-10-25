# Symmetric Quantization (Symmetric Fake-Quant)

## Problem

Symmetric quantization maps a floating-point tensor onto integer grid points (e.g. int8), with the quantization zero point fixed at 0 and the range symmetric about the origin. It is a core building block of quantization-aware training (QAT): **fake-quant** = run quantization → dequantization once through and return the result "still in floating point but contaminated by quantization noise", so gradients pass straight through.

## Rules

- Implement `solve(x)`, perform per-tensor symmetric quantization and return the **dequantized float tensor** (same shape and dtype as `x`):
  1. $\text{scale} = \dfrac{\max_i |x_i|}{127}$
  2. $q_i = \operatorname{clamp}(\operatorname{round}(x_i/\text{scale}),\,-127,\,127)$
  3. return $q_i \cdot \text{scale}$
- $\text{scale}$ is always positive; an all-zero tensor is returned unchanged (to avoid division by zero).
- The result must match `torch.fake_quantize_per_tensor_affine(x, scale, 0, -127, 127)` (with the same scale).

## Examples

```
Input: x = [0.2, -1.0, 0.7, -0.3]
      absmax = 1.0 → scale = 1/127 ≈ 0.0078740
      grid integers q = [25, -127, 89, -38]
Output: [0.19685, -1.0, 0.70079, -0.29921]
(-1.0 lands on the largest grid point and is kept exactly; 0.2 falls on grid point 25 → 25/127 ≈ 0.19685)
```

## Constraints

- Supports float32 / float64; output dtype matches the input
- `round` rounds to nearest (banker's rounding, consistent with `torch.round`)
- Do not use `fake_quantize_*`; hand-write the three steps of scale estimation + round/clamp/dequant
