# Diffusion: SeaCache Spectral-Indicator Gate Step (SEA Filter + Accumulated Budget + Residual Extrapolation)

## Problem

SeaCache (Spectral-Evolution-Aware Cache) uses a **spectral indicator** to decide whether consecutive video-diffusion steps changed little: the raw vision latent is passed through a three-axis Wiener-style spectral filter to form the SEA indicator, and if the **accumulated relative L1** of the indicator across steps stays within budget, the block stack is skipped. Implement `solve(vision_latent, gen_input, gen_output, state, step_index, sigma, num_inference_steps, threshold=0.25, residual_order=1, retention_steps=1, cache_end_steps=1, max_consecutive_cached=2, power_exp=3.0)`, three stages per step:

```text
# 1. SEA filter indicator (vision_latent (C,T,H,W) -> movedim(0,-1) -> (T,H,W,C))
spectrum = fftn(x, dims=(0,1,2));  sigma <- clamp(sigma, 1e-6, 1-1e-6)
clean_power = 1/(|f|^power_exp + 1e-16)                    # per spatial axis
axis_gain   = (1-σ)·cp / ((1-σ)²·cp + σ² + 1e-16)
gain = Π axis_gain;  gain <- gain / mean(gain)             # density normalization
indicator = ifftn(spectrum·gain).real

# 2. Gate (handle step adjacency first, then the budget)
non-adjacent (step ≠ last+1) -> full trajectory reset
distance  = mean|cur − prev| / (mean|prev| + 1e-16)        # shape/dtype mismatch -> inf
candidate = accumulated_distance + distance
compute   <=>  forced ∨ candidate >= threshold               # note: >=
forced (or): step < retention_steps, step >= N − cache_end_steps,
             previous_indicator is None, consecutive >= max_consecutive_cached (when >0)
compute steps zero accumulated; skip steps accumulated = candidate

# 3. Residual reuse
compute step: output = gen_output; history appends (step, gen_output − gen_input)
              truncated to residual_order+1; consecutive = 0
skip step:    output = gen_input + cached_residual; consecutive += 1
              with residual_order=1 and two distinct-step records, linear extrapolation:
              cached = latest + (latest − previous)·(step−latest_s)/(latest_s−prev_s)
skip decided but history empty / history residual shape or dtype mismatch
              -> fail-open compute with accumulated zeroed
```

Return `dict(output, computed, distance, accumulated_distance, state)` with a five-key state `{history, previous_indicator, accumulated_distance, consecutive_cached, last_step}` (new dict, functional).

## Key Semantics

- **The indicator is built in the frequency domain**: three-axis gains multiplied then normalized by their mean, `ifftn(...).real` back to the original dtype; σ is clamped to `[1e-6, 1−1e-6]` inside the filter (endpoint behaviour: with T=H=W=1 the normalized gain is exactly 1 and the indicator ≈ identity).
- **`>=`, not `>`**: candidate exactly equal to the threshold computes — with threshold=0 SeaCache never skips (zero-distance candidate=0 ≥ 0).
- **distance references the previous step's (filtered) indicator**, not the raw latent; the denominator is `mean|prev| + 1e-16`.
- **Non-adjacent steps** (forward or backward jump, anything ≠ last+1) reset history/previous_indicator/accumulated/consecutive first — after a reset the first observation forces compute.
- **The extrapolation scale can ≠ 1**: with non-adjacent history steps (e.g. steps 0 and 2), `step_scale = (3−2)/(2−0) = 0.5`.
- **Two fail-open channels**: a skip decision with empty history, or a history residual whose shape/dtype mismatches gen_input, both fall back to compute with accumulated zeroed.

## Judgement-domain Conventions

- The official gate memoizes (step, sigma) for the root+leader double entry within one forward; the judgement domain calls once per step explicitly, so the memo is omitted.
- The official raw_vision is a list of latents with per-element distance averaging; the judgement domain uses a single latent.
- The official und dual stream (Cosmos), FSDP/parallelism synchronization, and autograd/no-context fail-open branches are not implemented; the generic video leader path is taken.
- `threshold ≥ 0`, `residual_order ∈ {0,1}`, `retention_steps/cache_end_steps/max_consecutive_cached ≥ 0`, `sigma ∈ [0,1]`, `power_exp > 0` (validated inside sea_filter) are explicit judgement-domain checks.

## Example

6 steps, retention=1, cache_end=1, max=2, threshold=0.25: step 0 forced compute; step 1 tiny change (distance≈0.01) skips; step 2 large change (candidate≈0.3) computes and zeroes; step 3 identical skips; step 4 tiny change skips (consecutive=2); step 5 hits the cache-end window and forces compute. The computed sequence is `[T,F,T,F,F,T]`.

Data scale: latents `(1–2, 1–2, 1–4, 1–4)` float32, N ≤ 8, threshold ∈ [0, 10].
