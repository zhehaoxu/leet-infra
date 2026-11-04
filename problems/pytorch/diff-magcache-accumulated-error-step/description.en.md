# Diffusion: MagCache Accumulated-Error Gate Step (Retention Window + Accumulated-Error Skip)

## Problem

MagCache (Magnitude-aware Cache) lets a diffusion Transformer **skip entire blocks** late in denoising: unlike FBCache (764), which compares residuals step by step, it maintains an **accumulated error** — magnitude ratios multiply and error accumulates across consecutive skipped steps, and execution only resumes once the threshold is exceeded. Implement `solve(hidden_states, tail_output, state, step, num_inference_steps, mag_ratios, threshold=0.06, retention_ratio=0.2, max_skip_steps=3)`:

```text
retention_step = int(retention_ratio * num_inference_steps + 0.5)   # round-half-up
step < retention_step  ->  forced compute (no accumulation)

step >= retention_step (accumulate first, decide second):
    ratio *= mag_ratios[step]          # out-of-range falls back to 1.0
    steps += 1
    err   += abs(1.0 - ratio)          # accumulates |1-Πscale|, not per-step |1-scale|
    skip   <=>  previous_residual is not None
                and err   <= threshold
                and steps <= max_skip_steps
    not satisfied -> reset (ratio=1.0, steps=0, err=0.0) and compute
```

- **Compute path**: `output = tail_output` (the full execution result); `previous_residual = tail_output − hidden_states`.
- **Skip path**: `output = hidden_states + previous_residual` (the last compute's residual), accumulators **kept, not reset**.

Return `dict(output, computed, accumulated_err, state)` with a new state dict (functional), keys `{previous_residual, accumulated_ratio, accumulated_steps, accumulated_err}`.

When `mag_ratios` length ≠ `num_inference_steps`, apply the official `nearest_interp` nearest-neighbour interpolation: `scale = (src_len−1)/(target_len−1)`, `idx = round(grid·scale)` (torch.round banker's rounding), target==1 takes the last element.

## Key Semantics

- **Accumulate first, decide second**: the err accumulation happens **before** the decision — even when this step must compute (e.g. the first step has no residual), ratio/steps/err have already absorbed this step's contribution; only the compute branch resets. Skip steps keep the accumulators (across consecutive skips the ratio keeps multiplying and err keeps accumulating).
- **err accumulates |1−Πscale|**, not |1−scale|: two steps of 0.98 give err = 0.02 + |1−0.98²| = 0.0596, not 0.04.
- **Retention rounds half-up**: `int(ratio·N + 0.5)` — 0.2·10=2.0→2, 0.25·10=2.5→3, 0.2·8=1.6→2; the truncated `int(ratio·N)` differs whenever frac(ratio·N)≥0.5.
- **Stale residual**: the skip output uses the **last compute step's** residual, independent of the current tail_output.
- **Out-of-range scale falls back to 1.0** (official behaviour): step ≥ len(ratios) gives scale=1.0; the ratio stays unchanged but err still accumulates |1−ratio|.

## Judgement-domain Conventions

- The official hooks live on head/tail blocks sharing state, with step_index advanced by the tail hook; the judgement domain takes step as an explicit argument.
- The official calibrate mode (for measuring ratios) is not implemented; mag_ratios must be provided explicitly.
- The official text+image concatenation branch on skip (Flux/SD3 shape-mismatch specific) is not implemented; the judgement domain requires hidden_states and tail_output to share a shape.
- The four-key state whitelist (unknown keys reported before missing keys), `threshold ≥ 0`, `retention_ratio ∈ [0,1]`, `max_skip_steps ≥ 1` are explicit judgement-domain checks.

## Example

N=10, retention_ratio=0.2 → retention_step=2: steps 0/1 force compute. From step 2 with ratios=[0.98,…]: step 2 has no residual → compute+reset; step 3 ratio=0.98, err=0.02≤0.06 → skip (output = hidden + old residual); step 4 err=0.0596 still skips; step 5 err=0.1184>0.06 → compute+reset.

Data scale: tensors `(1, 2)`–`(2, 2)` float64, N ∈ [1, 10], threshold ∈ [0, 100].
