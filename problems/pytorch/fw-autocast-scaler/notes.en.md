# Explained: autocast + GradScaler Training Loop Accounting

## Origin

PyTorch AMP docs (*torch.amp*): the CPU default dtype of `torch.amp.autocast(device_type='cpu')` is `bfloat16`, and `dtype=torch.float16` can be passed explicitly; matmul/linear sit on the cast list (demoted to half precision), while a parameter's `.grad` is converted back to the parameter dtype (fp32) at the cast boundary. `torch.amp.GradScaler(device='cpu', enabled=True)` works on CPU in recent versions (empirically `is_enabled()==True` on the judge environment's torch 2.14; older versions warn and auto-disable on CPU — behavior is version-dependent, so the statement's numbers come from actually running the reference solution). State-machine semantics: `unscale_` may be called only once per update cycle — a second call raises `RuntimeError: unscale_() has already been called on this optimizer since the last update()`; `step` skips when it sees inf, `update` then backs off by `backoff_factor`, otherwise `_growth_tracker += 1`, and after `growth_interval` consecutive successes the scale is multiplied by `growth_factor` and the tracker resets.

## Key points and pitfalls

1. **The skip signal is in the weights, not the return value**: the new `scaler.step()` returns `None` whether or not it skipped; only `torch.equal(before, after)` (or the subsequent scale change) serves as ledger evidence.
2. **Scale only guards underflow; the math is untouched**: after `unscale_` the gradients return to true values, so the weight trajectory is bitwise identical to pure fp32 training (given binary-exact values — the statement uses dyadic decimals throughout).
3. **The scale trajectory is a state-machine problem**: `[16,8,8,16,8,8]` is fully determined by "where the infs are injected + growth_interval=2"; the oracle replays the same theorem in ten lines of pure Python without importing GradScaler.
4. **CPU autocast dtype rules**: `x@w` outputs fp16 and `mean` stays fp16 (the CPU promotion list differs from CUDA's), yet `w.grad` is fp32 — "mixed precision" mixes forward compute precision with backward storage precision.
5. **Hard protection against ordering violations**: besides unscale→unscale, real-world accidents (clipping without unscale, stepping twice before update) are also caught by this state machine; the probe here is the most common instance.

## Cross-check basis

The solution runs the real autocast/GradScaler; `_oracle` replays everything in pure Python floats: per step it hand-computes \(out=1.5w_0-0.5w_1\), \(loss=(out-1)^2\), the analytic gradient \(\nabla=2(out-1)x\), and updates scale/tracker by the state-machine rules — the forward-independent "frozen weights on skip" fact is checked directly via the `updated` bool ledger. Three more invariants: the loss repeats in place after a skip, the t3 doubling ledger, and double-unscale must raise.

## Complexity

6 steps × 2 parameters — microsecond level; same for the oracle.
