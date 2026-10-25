# Asymmetric Quantization (Asymmetric Fake-Quant)

## Problem

Asymmetric quantization calibrates with the data's $\min/\max$, aligning the integer range $[0,255]$ with the float range $[\min,\max]$, so that float 0 falls on some non-zero integer grid point (the zero_point). Compared with the symmetric version, it fully uses the grid points and is better suited to tensors whose **distribution is not symmetric about 0** (such as all-positive activations after ReLU), at the cost of storing an extra zero_point.

## Rules

- Implement `solve(x)`: per-tensor uint8 asymmetric quantization returning a **dequantized float tensor** (same shape and dtype as `x`):
  1. $\text{scale} = \dfrac{\max - \min}{255}$
  2. $z = \operatorname{clamp}(\operatorname{round}(-\min/\text{scale}),\,0,\,255)$
  3. $q = \operatorname{clamp}(\operatorname{round}(x/\text{scale}) + z,\,0,\,255)$
  4. Return $(q - z)\cdot\text{scale}$
- When $\text{scale}=0$ (constant tensor), return the input unchanged.
- The result must match `torch.fake_quantize_per_tensor_affine(x, scale, z, 0, 255)` (with the same scale/z).

## Examples

```
Input: x = [-1.0, 0.0, 0.5, 2.0]
      min=-1, max=2 → scale = 3/255 ≈ 0.011765, zero_point = 85
Output: [-1.0, 0.0, 0.49412, 2.0]
(min/max hit the grid boundaries and are restored exactly; 0.5 lands on grid point 42 → (42+85-85)*scale ≈ 0.49412)
```

## Constraints

- Support float32 / float64; the output dtype matches the input
- `round` uses round-half-to-even (banker's rounding, matching `torch.round`)
- Do not use `fake_quantize_*`; estimate scale/zero_point and do round/clamp/dequant by hand
