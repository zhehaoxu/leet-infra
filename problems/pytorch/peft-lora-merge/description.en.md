# PEFT: LoRA Weight Merging (merge_and_unload)

## Problem

After LoRA training, deployment usually "folds" the low-rank side path back into the original weight to obtain an equivalent dense linear layer — this is `merge_and_unload()` in the PEFT library: \(W' = W_0 + \frac{\alpha}{r} B A\). After merging, inference needs only one matrix multiplication with no extra side-path overhead, and this layer's \(\Delta W\) can also be quantized/distilled/moved to another device.

This problem asks you to implement this merge: given the frozen weights and the low-rank factors, produce the merged weight matrix \(W'\). It is element-wise equivalent to the "LoRA step-by-step forward" (for the same input \(x\), \(xW'^\top = xW_0^\top + \frac{\alpha}{r}(xA^\top)B^\top\)).

## Rules

Given `W0` \((d_\text{out}, d_\text{in})\), `A` \((r, d_\text{in})\), `B` \((d_\text{out}, r)\), scalars `alpha` and `r`:

1. Scaling factor \(s = \alpha / r\).
2. Low-rank update \(\Delta W = B A\) (shape \((d_\text{out}, d_\text{in})\)).
3. Merged weight \(W' = W_0 + s\,\Delta W\).

Return `W'` (shape \((d_\text{out}, d_\text{in})\), the same shape as `W0`). float32 throughout, ready-made PEFT libraries forbidden.

## Examples

Reusing \(W_0\) (3×5), \(A\) (2×5), \(B\) (3×2) from the LoRA forward problem, \(\alpha=8.0\), \(r=2\) (\(s=4.0\)):

```
W' = [[ 0.780, -0.560,  0.100,  0.540,  0.040],
      [ 0.560,  0.280,  0.140,  0.000,  1.200],
      [-0.740,  0.780,  0.580, -0.280, -0.280]]
```

## Constraints

- CPU float32; `W'` has the same shape as `W0`.
- For any input `x`, `x @ W'.T` must match the LoRA step-by-step forward element-wise.
- With `alpha=0`, `W'` is always equal to `W0` (identity folding without fine-tuning).
