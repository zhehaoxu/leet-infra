# Diffusion: TaylorSeer Cache Step (Gating + Divided Differences + Taylor Extrapolation)

## Problem

TaylorSeer (arXiv 2503.06923) lets diffusion-Transformer modules **skip the forward pass** on some denoising steps and approximate the output via Taylor extrapolation. Implement `solve(outputs, state, step, cache_interval=5, disable_cache_before_step=3, disable_cache_after_step=None, max_order=1)`:

**Gating** (decides compute vs predict for this step):

```text
should_compute = (step < disable_cache_before_step)                     # warmup
              or ((step − disable_cache_before_step − 1) % cache_interval == 0)
              or (disable_cache_after_step is not None and step >= disable_cache_after_step)
```

**Compute steps** run the divided-difference recursion (delta = step − last_update_step):

```text
f[0] = output
f[j+1] = (f[j]_new − f[j]_prev) / delta        # j = 0..max_order−1
```

The first update stores only f[0]; if f[j]_prev is missing or **changes shape**, break (restart the expansion with order 0 only); delta == 0 raises.

**Predict steps** extrapolate (offset = step − last_update_step):

```text
output = Σ_j (offset^j / j!) · f[j]            # j over existing orders
```

Return `dict(output, computed, state)` with a new state dict (functional, input unmodified).

## Key Semantics

- **Warmup steps also update**: steps with `step < disable_cache_before_step` force compute and update the factors (official comment: "factors are updated, caching not applied") — without them the first predict after warmup would hit empty factors.
- **The interval off-by-one**: compute steps satisfy `(step − before − 1) % interval == 0`. With before=3, interval=5 they are 0,1,2 (warmup) + 4, 9, 14…, while step 3 is a predict step ((3−4)%5=4≠0).
- **delta is the compute-step gap** (4→9 gives 5), not constant 1 — the differences must be divided by it to be per-step derivatives.
- **offset is the predict-step distance since the last compute** (varying 1..interval−1), not constant 1.
- **Shape-change restart**: when the output shape changes between computes (e.g. Qwen-Image KV prefill), old factors cannot be differenced and the expansion restarts at order 0.
- **Inconsistent state**: `last_update_step is None` **or** empty factors both make predict raise "Cannot predict without prior initialization/update." (the official raise condition is an or).

## Judgement-domain Conventions

- The official hook increments `state.current_step` internally; the judgement domain takes step explicitly and the state carries no current_step.
- The official code supports tuple outputs; the judgement domain uses a single tensor. The official factor dtype defaults to bfloat16; the judgement domain fixes float64.
- `cache_interval ≥ 1` (official 0 would ZeroDivisionError) and `max_order ≥ 1` are explicit judgement-domain checks.
- With before=0, step 0 falls into predict with no initialization — an officially reachable path; the raise branch is kept.

## Example

before=3, interval=5, max_order=2, after=None over the first 12 steps: compute steps = 0,1,2,4,9; predict steps = 3,5,6,7,8,10,11. Step 3 has offset=3−2=1; step 8 has offset=8−4=4. With cooldown after=6, steps 6/7/8 also force compute.

Data scale: output tensors `(1, 2)`–`(2, 2)` float64, factor order ≤ 3.
