# Diffusion: PAB Attention Broadcast Step (Three-Way Dispatch + Iterative Broadcast Gate)

## Problem

Pyramid Attention Broadcast (PAB) skips attention computation by exploiting the high similarity of attention states across adjacent timesteps: cross-attention is the most similar (skipped most often), temporal next, spatial last. Implement `solve(name, is_cross_attention, hidden_states, state, current_timestep, *_block_skip_range, *_timestep_skip_range, *_block_identifiers)`, two stages per step:

```text
# 1. Attention-layer dispatch (spatial > temporal > cross priority, re.search any-hit)
spatial  = match(spatial_ids)  and spatial_range  is not None and not is_cross
temporal = match(temporal_ids) and temporal_range is not None and not is_cross
cross    = match(cross_ids)    and cross_range    is not None and is_cross
no match -> no hook (always compute, state passed through)
all three block_skip_ranges None -> official fallback spatial_attention_block_skip_range=2

# 2. Broadcast gate
within         = lo < current_timestep < hi          # open interval (strict)
should_compute = cache is None or not within or iteration % block_skip_range == 0
compute   -> output = the real attention output
broadcast -> output = cache (the last output)
state.cache is updated to this step's output on every call; iteration +1 per call
```

Return `dict(output, computed, block_type, state)` with a two-key state `{iteration, cache}` (new dict, functional; unmatched layers pass the state through unchanged).

## Key Semantics

- **Priority is semantics**: the official default spatial_ids contain `"blocks"`, and `re.search("blocks", "temporal_transformer_blocks.0.attn1")` hits — a temporal layer name is claimed by spatial first; temporal only gets its turn when spatial_range is None. Cross layers are excluded from spatial/temporal by the `is_cross_attention` flag.
- **Open interval**: `lo < t < hi` strictly — t equal to lo or hi is outside the range and always computes (both endpoints of the official default (100, 800) count as outside).
- **Modulo broadcast**: calls 0, N, 2N… compute, the rest broadcast; iteration increments on every call (broadcasts included).
- **The cache updates every step**: compute steps cache this step's output; broadcast steps cache this step's output (the old cache) — when the timestep leaves the range and compute resumes, the cache refreshes immediately.
- **The official `iteration == 0` branch is dead code** (fully subsumed by `0 % n == 0`) and is not included in the judgement domain.

## Judgement-domain Conventions

- The official hook attaches per attention module with current_timestep from a callback; the judgement domain takes the layer name / is_cross / current timestep explicitly, merging dispatch and gate into one function.
- The official code validates nothing; the judgement domain adds explicit checks for the name string, is_cross bool, tensor, two-key state, block_skip_range ≥ 1 or None, timestep-range integer pairs, and identifier string sequences.
- The official broadcast stores a reference; the judgement domain stores a clone (functional).

## Example

A spatial layer with block_skip_range=2, inside the range at t=500: the computed sequence is `[T,F,T,F,T]`; the 2nd call outputs the 1st call's output (cached), and the 3rd (iteration=2, 2%2==0) resumes compute. At t=100 (exactly lo) or t=800 (exactly hi) the sequence is `[T,T,T,T,T]`. With block_skip_range=3 it is `[T,F,F,T,F]`.

Data scale: tensors `(1, 2)` float32, N ≤ 8, arbitrary integer timestep ranges.
