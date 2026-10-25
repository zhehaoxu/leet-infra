# Explained: Training Resume Continuity Ledger

## Origin

PyTorch's *Tips for Reproducibility* and standard checkpointing practice: bitwise-identical resume requires saving/restoring four things — model, optimizer, scheduler, and `torch.get_rng_state()` (plus CUDA RNGs on GPU). The aliasing behavior lives in `torch/optim/optimizer.py`: `Optimizer.state_dict()` returns tensors that reference the live buffers, and `load_state_dict` binds the passed tensors straight into the new optimizer with no copy on CPU — so loading one ckpt twice lets the second run's in-place `opt.step()` corrupt what the first run reads. `torch.Tensor.to(device)` likewise returns the same object when the device already matches.

## Key points and pitfalls

1. **All four ledgers are mandatory**: even with weights, momentum, and `last_epoch` correct, skipping the RNG restore shifts every later \((x,y)\) draw, so losses diverge from step 4 (here 0.0067 → 1.0938). Path 3 is precisely this missing-ledger control experiment.
2. **Optimizer state is an alias, not a snapshot**: `o.state_dict()["state"][0]["momentum_buffer"]` *is* the live buffer; after `o2.load_state_dict(sd)` the new optimizer holds those same objects. Each resume must deep-copy per tensor (the reference solution clones once at save and once at load); otherwise a later path mutates the shared buffer and both `momentum_restored` and `losses_no_rng` get mis-recorded.
3. **Snapshot timing**: clone the final weights and momentum at the end of path 2 *before* path 3 starts; reading them after all paths run sees the corrupted shared tensors — a real bug the first version of this reference solution hit.
4. **The lr ledger**: with `sched.step()` at the loop tail, step \(i\) uses \(lr=0.1\cdot0.5^{\lfloor i/2\rfloor}\) (StepLR reads `last_epoch`; `__init__` has already previewed slot 0).
5. **`resumed_bitwise_identical` uses list `==`**: every loss is `float(loss.item())`, and the same IEEE operation sequence reproduces bitwise in Python; allclose would mask the real misalignment.
6. **Module `state_dict()` does not have the aliasing problem**: `load_state_dict` writes into nn.Module parameters with `copy_`, so module save/load is safe; only optimizer/scheduler state swaps references.

## Cross-check basis

The solution runs the three real paths. The `_oracle` is pure Python: it replays the \((x,y)\) data stream from the same seed (torch's Philox stream cannot be reproduced in numpy — the draws are the single primitive shared with the framework), then hand-computes everything in float64: forward \(z=W_1x+b_1,\ a=\tanh z,\ \hat y=w_2\cdot a+b_2\); gradients \(\partial L/\partial \hat y=2(\hat y-y)\) and \(1-a^2\); SGD momentum \(b\leftarrow0.9b+g,\ w\leftarrow w-lr\cdot b\); \(lr=0.1\cdot0.5^{\lfloor i/2\rfloor}\). Losses use `allclose(1e-5,1e-5)`; additionally `losses_resumed == losses_full`, the first three entries bitwise-equal, tail differences \(>10^{-3}\) elementwise, and exact assertions on booleans and `last_epoch`.

## Complexity

\(O(T\cdot d)\) with \(T=6\) steps on 2-dimensional samples — milliseconds; memory is only a few tiny tensors.
