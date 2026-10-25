# Training Resume Continuity Ledger

## Problem

The acceptance criterion for "resume from checkpoint" is not that the loss curve looks similar — it is **bitwise identity**: save all four ledgers (model, optimizer, scheduler, global RNG), resume from step 3, run 3 more steps, and the loss sequence must match an uninterrupted 6-step run exactly. Skip any ledger (especially the RNG) and the second half of the data stream shifts wholesale. Worse, `Optimizer.state_dict()` returns tensors that alias the live momentum buffers, and `load_state_dict` binds the passed tensors straight into the new optimizer (no copy on CPU), so loading the same ckpt twice lets the two runs corrupt each other in place. This problem runs three paths and turns resume continuity, missing-RNG divergence, and ckpt aliasing into checkable ledgers.

## Rules

Implement `solve()`, run the three paths from the starter docstring, and return the continuity ledger dict:

1. **Path 1 (golden trajectory)**: after `torch.manual_seed(1234)` build a fresh model / SGD(lr=0.1, momentum=0.9) / StepLR(step_size=2, gamma=0.5) and run 6 uninterrupted steps → `losses_full`.
2. **Path 2 (full restore)**: run 3 steps on the same stream, save the five-item ckpt (model/opt/sched/rng/step, where the optimizer state must be a per-tensor deep copy), burn 2 unrelated randoms via `torch.randn(2)`, then load the ckpt into brand-new objects, call `set_rng_state`, and continue steps 3..6 → `losses_resumed`.
3. **Path 3 (RNG not restored)**: structurally identical to path 2, same junk `torch.randn(2)` burn, same ckpt loads, but no `set_rng_state` → `losses_no_rng`.

The single-step order is fixed as `zero_grad → randn draws → loss → backward → opt.step() → sched.step()`, and step \(i\) uses \(lr=0.1\cdot 0.5^{\lfloor i/2\rfloor}\). Compare `losses_resumed` to `losses_full` with Python list `==` (bitwise); numeric losses are matched with `atol=1e-5, rtol=1e-5`.

## Examples

```
Input: none
Output:
{
 "losses_full": [0.5204203128814697, 0.6624248623847961, 0.0687742680311203,
                 0.0066848150454461575, 0.05954299122095108, 0.5288273692131042],
 "losses_resumed": [0.5204203128814697, 0.6624248623847961, 0.0687742680311203,
                    0.0066848150454461575, 0.05954299122095108, 0.5288273692131042],
 "losses_no_rng": [0.5204203128814697, 0.6624248623847961, 0.0687742680311203,
                   1.0938212871551514, 0.1766050010919571, 1.190037727355957],
 "resumed_bitwise_identical": true,
 "no_rng_tail_diverges": true,
 "final_weight_identical": true,
 "momentum_restored": true,
 "sched_last_epoch_resumed": 6
}
```

The first three entries of `losses_no_rng` match `losses_full` bitwise (same stream before the save); divergence starts at the fourth — the 2 junk randoms shift every later \((x,y)\).

## Constraints

- Each path starts its stream from `torch.manual_seed(1234)`; losses are float32 `loss.item()` compared with `atol=1e-5, rtol=1e-5`; booleans and `sched_last_epoch_resumed` must match exactly.
- `resumed_bitwise_identical` is Python list `==` (bitwise), not allclose.
- If the saved/loaded optimizer state is not deep-copied per tensor, path 3 reads momentum buffers already mutated in place by path 2, corrupting both `losses_no_rng` and `momentum_restored`.
