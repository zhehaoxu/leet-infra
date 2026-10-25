# FP16 Dynamic Loss Scaling: Single Step

## Problem

The smallest normal number FP16 can represent is about $6\times10^{-5}$, and shallow-layer gradients are often smaller → they **underflow to 0** during backpropagation, and training stalls. The solution: before the backward pass, multiply the loss by a large integer `loss_scale` (lifting the gradients as a whole into the representable range of fp16), and divide it back before the update; as soon as overflow (inf/NaN) is detected, skip this step and reduce the scale; if several consecutive steps do not overflow, increase the scale. This is the core logic of `torch.cuda.amp.GradScaler`. This problem implements the complete decision for **one optimizer step**.

## Rules

Implement `solve(grads, loss_scale, growth_interval, consec_ok, backoff_factor=2.0, growth_factor=2.0)`:

- `grads`: fp32 gradient tensor (the true gradients before loss scaling).
- `loss_scale`: the current scaling factor (float, > 0).
- `growth_interval`: after how many consecutive non-overflow steps the scale is increased.
- `consec_ok`: the number of **consecutive non-overflow** steps up to this step (external state of the scaler).

Single-step flow (implement each item exactly):

1. `scaled = (grads * loss_scale)`, then convert to **fp16**.
2. `overflow =` whether **inf or NaN appears** in the converted fp16 (either `isinf | isnan` true). The fp16 maximum is 65504; exceeding it gives inf.
3. If `overflow`: `skip_update=True`; `new_loss_scale = loss_scale / backoff_factor`; `new_consec_ok = 0`; `unscaled_grads = grads` (no update this step; return as is, without unscaling).
4. Otherwise `skip_update=False`:
   - Let `c = consec_ok + 1`. If `c >= growth_interval`: `new_loss_scale = loss_scale * growth_factor`, `new_consec_ok = 0`; otherwise `new_loss_scale = loss_scale`, `new_consec_ok = c`.
   - `unscaled_grads = (scaled_fp16.float() / loss_scale)` converted back to `grads.dtype` (quantized into fp16 first and then divided back, **introducing fp16 rounding error**—this is precisely the precision loss this problem is meant to illustrate).

Return a **dict**: `{"overflow": bool, "skip_update": bool, "new_loss_scale": float, "new_consec_ok": int, "unscaled_grads": torch.Tensor}`.

## Examples

```
grads = [0.001, 0.002], loss_scale = 65536, growth_interval = 2000, consec_ok = 0
  scaled → fp16 does not overflow
  overflow=False, skip_update=False, new_loss_scale=65536, new_consec_ok=1
  unscaled_grads ≈ [0.00100040435791, 0.00200080871582]   # tiny distortion from the fp16 round trip

grads = [1.0, 2.0, 100.0], loss_scale = 65536, ...
  1.0*65536=65536 > 65504 → inf, overflow=True
  skip_update=True, new_loss_scale=32768.0, new_consec_ok=0, unscaled_grads=[1.0,2.0,100.0] (unchanged)

grads = [0.001,0.002], loss_scale=65536, growth_interval=2, consec_ok=1
  no overflow and c=consec_ok+1=2 ≥ 2 → growth triggered: new_loss_scale=131072.0, new_consec_ok=0
```

## Constraints

- The fp16 overflow criterion looks only at inf/NaN of `scaled.to(float16)`; do not approximate with `grads*loss_scale > some threshold` (the threshold varies with dtype, and the fp16 max is exactly 65504)
- `new_loss_scale` is a python float; `new_consec_ok` is an int; `unscaled_grads` has the same dtype as `grads`
- Do not modify the input `grads`
