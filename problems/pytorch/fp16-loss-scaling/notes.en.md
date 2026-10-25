# FP16 Dynamic Loss Scaling Explained

## Overflow and Underflow Are Two Different Things

fp16 has a **narrow representable range** (normal numbers roughly $6\times10^{-5}$ to 65504). Two things go wrong in mixed precision:

- **Loss backprop underflow**: small gradients directly become 0 (the most fatal; shallow layers cannot learn). Loss scaling solves this—multiply by a large number to lift the gradients into the representable range.
- **Scaled loss overflow**: after multiplying by the scale, values above 65504 → inf, contaminating everything downstream. So inf/NaN must be detected and the step skipped.

This problem only covers **overflow detection + scale adaptation**. "Lifting underflow" is the benefit, "possible overflow" is the cost, and the scaler dynamically finds the largest usable scale between the two.

## Why Not Unscale When Overflow Occurs

Once `scaled` produces inf, converting back (÷loss_scale) is still inf/NaN, and updating the weights with it would destroy the model. The standard practice is: **invalidate this step** (skip the optimizer step), only do `loss_scale /= backoff`, and try again with a smaller scale next round. Hence in this problem the overflow branch returns `unscaled_grads = grads` unchanged (it will not be used; this is only for contract completeness).

## The "Consecutive" Counter for Growth

`consec_ok` is external state; it is incremented by 1 each step and **reset to zero** on overflow. The scale is doubled and the counter reset only after reaching `growth_interval` (default 2000) consecutively. The logic belongs in the "no overflow" branch, and you should `+1` first and then compare `>=`; do not reverse the order (one step too few or too many causes long-run scale drift).

## fp16 Round-Trip Error Is a Feature, Not a Bug

When there is no overflow, `unscaled = fp16(grads*scale).float()/scale` differs slightly from the original `grads` (example `[0.001→0.001000404…]`), because the value is "quantized" into fp16's sparse grid points and back. Real training tolerates this error—in exchange for rescuing small gradients from underflowing to 0. The judge **requires** reproducing this fp16 rounding (not returning grads unchanged).

## Trust the dtype, Not a Threshold

The fp16 max is exactly 65504 (`torch.finfo(float16).max`). `1.0*65536=65536>65504` gives inf. Using the `isinf|isnan` criterion instead of a self-chosen threshold is both accurate and generalizable to bf16/other dtypes (bf16 has the same range as fp32 and almost never overflows—this is exactly why bf16 does not need loss scaling; see the extension).

## Related

- Same family: the quantization "precision tier" topic—`fp8-quantize` (a more extreme low bit width), `per-channel/group-wise-quantize` (inference quantization).
- Extension: bf16 has fp32's exponent range and usually does not need loss scaling; fp8 training needs per-tensor scaling (merging with the quantization family).
- Judging: `check_example` uses dumped fp16 round-trip values; there are also overflow/backoff, growth trigger, fp16 max boundary, and input-not-modified test cases.
