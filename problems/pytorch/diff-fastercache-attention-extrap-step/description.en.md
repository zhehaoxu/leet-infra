# Diffusion: FasterCache Attention Extrapolation Step (Batch Dance + Two-Slot Linear Extrapolation)

## Problem

FasterCache goes one step beyond PAB's "reuse the cache": on broadcast it does not reuse the old attention output directly but **linearly extrapolates** from the two most recent outputs, `t_1 + (t_1 − t_2)·weight`, coordinated with the denoiser-side unconditional-branch skip to form a batch dance. Implement `solve(attention_output, state, current_timestep, block_skip_range=2, timestep_skip_range=(-1, 681), weight=0.5, is_guidance_distilled=False)`:

```text
# 1. Skip gate
within = lo < current_timestep < hi                 # open interval (strict)
outside -> no skip
inside  -> skip = iteration % block_skip_range != 0  # iterations 0, N, 2N... compute
skip recheck: really skip only when is_guidance_distilled or state.batch_size != current batch
             (the cache holds only conditional halves; a full-batch forward must really run)

# 2. Linear extrapolation (on skip)
t_2, t_1 = cache  # the two most recent conditional outputs
if a cache entry's size(0) != current batch, slice the conditional half [batch:] first
output = t_1 + (t_1 - t_2) * weight                # otherwise output = attention_output

# 3. Cache maintenance
cache_output = output
if not distilled and output.size(0) == state.batch_size -> chunk(2, dim=0)[1]
cache = [cache[1], cache_output]                   # first step [c, c]; iteration += 1
```

Return `dict(output, computed, state)` with a three-key state `{iteration, batch_size, cache}` (new dict, functional; batch_size recorded on the first call and kept).

## Key Semantics

- **The interval only "permits" broadcasting**: the endpoints t==lo/t==hi (official default −1/681) are outside, and even conditional-half calls always compute there.
- **The batch-mismatch recheck is a second gate**: inside the range + modulo hit is still not enough — the cache holds only conditional halves, so a full-batch call (batch == state.batch_size) must really run. Distilled models (no uncond branch) are exempt.
- **The extrapolation references the two previous conditional outputs**: a cache entry accidentally carrying a full batch (2B) is sliced `[batch:]` first (conditional in the latter half), not `[:batch]`.
- **Two-slot rolling**: new `cache = [old cache[1], new conditional output]` — the extrapolation at call N+2 references call N's output and call N+1's extrapolated approximation, not older entries.
- **The half-batch condition on cache writes**: only when not distilled and `output.size(0) == state.batch_size` (full batch) does the write chunk the latter half; conditional-half calls (batch mismatch) enter the cache as-is.
- **The official `iteration > 0 and ...` is dead code** (0%N==0 always true) and is not included in the judgement domain.

## Judgement-domain Conventions

- The official hook attaches to attention modules with weight/current_timestep from callbacks; the judgement domain takes them explicitly (weight default 0.5, the official callback-less default).
- The official dispatch layer (identifier matching) is the same family as 767 PAB and is not repeated; skip ranges are explicit arguments.
- The official cache supports tuple multi-output branches; the judgement domain uses a single tensor.
- The official code validates nothing; the judgement domain adds tensor/range/weight/state three-key checks.

## Example

The full non-distilled dance (full batch 2B=4, conditional half B=2, N=2, inside the range, w=0.5): iter0 full-batch compute (cache takes the latter half ca); iter1 conditional-half skip (extrapolation = ca, both slots equal); iter2 full-batch compute (cache=[ca, cb]); iter3 conditional-half skip (extrapolation = cb + (cb−ca)·0.5); iter4 full-batch compute (cache=[approx3, cd]); iter5 skip referencing approx3, not cb. The computed sequence is `[T,F,T,F,T,F]`.

Data scale: tensors `(2, 4)`–`(4, 4)` float32, N ≤ 8, weight ∈ [0, 2].
