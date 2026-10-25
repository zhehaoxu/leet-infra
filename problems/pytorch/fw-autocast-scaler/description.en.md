# autocast + GradScaler Training Loop Accounting

## Problem

Mixed-precision stability hangs on two things: autocast decides which dtype each operator runs in, and GradScaler decides "which step gets skipped and how the scale rises and falls" on overflow. The order `scale → backward → unscale_ → step → update` is part of the state machine — misordering raises directly. This problem runs 6 steps on CPU with `torch.amp.autocast('cpu', dtype=torch.float16)` + `torch.amp.GradScaler` (injecting inf gradients at steps 1 and 4 to simulate overflow) and asks for the full ledger: dtypes, the scale trajectory, the skip count, and the weight trajectory.

## Rules

Implement `solve()`, execute the fixed setup and the 8-step loop order from the starter docstring, and return the ledger dict. Key points:

1. Every value is a binary-exact small decimal, so the fp16 forward has no rounding — the loss/weight trajectories can be recomputed step by step in plain floats.
2. The scale state machine: inf found → skip the update, \(scale\times0.5\), growth tracker reset to 0; consecutive successful steps driving the tracker to `growth_interval=2` trigger \(scale\times2\) and a reset.
3. Calling `unscale_` twice within one cycle must raise `RuntimeError` (the ordering-violation probe of step 4); catch it and continue normally.
4. "Was the weight actually updated" is decided by `torch.equal` around `step` — on CPU the return value of `scaler.step()` is not a skip signal.
5. Record the three dtype strings `out/loss/grad`: under autocast the matmul output is fp16, yet `w.grad` is fp32 (the autocast cast boundary restores gradients to the parameter dtype, and the scaling multiply keeps fp32).

## Examples

```
Input: none
Output:
{
 "out_dtype": "torch.float16", "loss_dtype": "torch.float16",
 "grad_dtype": "torch.float32",
 "scale_traj": [16.0, 8.0, 8.0, 16.0, 8.0, 8.0],
 "trackers": [1, 0, 1, 0, 0, 1],
 "updated": [true, false, true, true, false, true],
 "losses": [0.140625, 0.0087890625, 0.0087890625, 0.00054931640625,
            3.4332275390625e-05, 3.4332275390625e-05],
 "n_skipped": 2,
 "final_w_flat": [0.72412109375, 0.17529296875],
 "double_unscale_raises": true
}
```

Deriving t3: the skip at t1 freezes the weights at their post-t0 values, and scale 8 is in use from t2; the two consecutive successes t2, t3 push the tracker from 1 to 2, triggering the doubling back to 16 and a reset.

## Constraints

- `scale_traj/trackers/updated/booleans/dtype strings` compared exactly; numerics with `atol=1e-5, rtol=1e-5`.
- Fixed hyperparameters and injection steps; no randomness.
