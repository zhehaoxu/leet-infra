# Per-Channel Quantization (Per-Channel Fake-Quant)

## Problem

Per-tensor symmetric quantization (see `symmetric-quantize`) uses **one** scale for the entire tensor. Convolution/linear weights often have wildly different dynamic ranges across output channels — a single outlier channel inflates the scale and wipes out the resolution of all the other channels. **Per-channel (per-axis) quantization** gives every channel its own scale and is the de facto standard for weight quantization. This problem implements the fake-quant version of it.

## Rules

Implement `solve(x, axis)` to perform symmetric per-channel fake-quant along `axis` and return the **dequantized floating-point tensor** (same shape and dtype as `x`):

1. For each channel $c$ (an index along `axis`), compute $\text{absmax}_c = \max_{i \notin \text{axis}} |x|$ over **all remaining dimensions** of that channel.
2. $\text{scale}_c = \text{absmax}_c / 127$.
3. $q = \operatorname{clamp}(\operatorname{round}(x / \text{scale}_c),\,-127,\,127)$, and return $q \cdot \text{scale}_c$.
4. **Degenerate channel**: if a channel is all zeros ($\text{absmax}_c = 0$), that channel returns 0 unchanged (avoids division by zero, and it must match the built-in torch oracle).

> `axis` is the "channel dimension". The scale vector has length `x.size(axis)` and is broadcast over the remaining dims. For example, weights `[out, in]` with `axis=0` → one scale per `out` row, with absmax reduced along the `in` dim.

## Examples

```
x    = [[1.0, -4.0, 2.0, 0.5],
        [0.0,  0.0, 0.0, 0.0],
        [3.0,  1.5,-6.0,-0.75]],  axis = 0

per-channel absmax = [4.0, 0.0, 6.0]  →  scale = [4/127, 0, 6/127]
output = [[1.00787, -4.0, 2.01575, 0.50394],
        [0.0,      0.0, 0.0,     0.0     ],     # all-zero channel stays 0
        [3.02362,  1.51181, -6.0, -0.75591]]
```

(Row 0 has absmax=4, which lands exactly on a grid point, so `-4.0` is restored exactly; row 2: `3.0→round(3/scale)=80→80*scale≈3.0236`)

## Constraints

- Support arbitrary-dim `x` (≥1 dim) and any valid `axis` (may be negative); `round` uses round-half-to-even (banker's rounding, same as `torch.round`)
- The result must match `torch.fake_quantize_per_channel_affine(x, scales, zeros, axis, -127, 127)` (`scales = absmax/127`, `zeros = 0` int32) — this is the built-in oracle for this problem
- Do not use `fake_quantize_*`; hand-write absmax/scale/round/clamp/dequant
