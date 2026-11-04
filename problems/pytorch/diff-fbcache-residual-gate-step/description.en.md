# Diffusion: First Block Cache Residual Gate Step (Relative-L1 Decision on the First-Block Residual)

## Problem

First Block Cache (FBCache, ParaAttention) lets a diffusion Transformer **skip every block after the first** when consecutive denoising steps change little. Implement `solve(head_input, head_output, tail_output, state=None, threshold=0.05)`:

```text
residual = head_output − head_input                   # first-block residual
diff     = absmean(residual − prev_residual) / absmean(prev_residual)
compute  <=>  prev_residual is None or diff > threshold   # first step forces compute
```

- **Compute path**: all blocks really run, `final_output = tail_output`; state updates `head_block_output = head_output`, `head_block_residual = residual`, `tail_block_residuals = tail_output − head_output`.
- **Cache path** (`diff <= threshold`): intermediate blocks pass through, `final_output = tail_block_residuals(cached) + head_output`; **all three state entries keep their old values** (consecutive cache steps compare against the last compute step's residual — stale comparison).

Return `dict(final_output, computed, diff, state)` with a new state dict (functional).

## Key Semantics

- **Both numerator and denominator are absmean (mean of absolute values)** — not mean, not max: the residual difference can be entirely negative (residual shrinking), and without abs the negative diff would wrongly cache.
- **Stale comparison**: cache steps update no state entry; across consecutive cache steps the diff reference is always the **last compute step's** residual.
- **The cache output formula**: `cached tail residual + current head output` — not `+ head_input` (the head output ≠ input), nor the current tail_output.
- **Zero-denominator normalization**: `absmean(prev) == 0` yields 0/0=nan (nan>x is False → cache) or x/0=inf (→ compute) officially; the judgement domain normalizes explicitly to 0.0/inf with the same semantics.
- **First step / post-reset**: `head_block_residual is None` forces compute and rebuilds the tail residual; a cache decision with `tail_block_residuals is None` is rejected explicitly (the official code would TypeError).

## Judgement-domain Conventions

- The official hooks live on head/middle/tail blocks sharing a StateManager; the judgement domain compresses one forward into `(head_input, head_output, tail_output)` plus an explicit state — the intermediate blocks' pass-through is an identity mapping on the cache path, mathematically equivalent.
- The official code supports tuple dual outputs (hidden_states/encoder_hidden_states); the judgement domain uses a single tensor.
- `threshold ≥ 0` is an explicit judgement-domain check (the official default 0.05 has none).

## Example

prev residual `(0.5, 0.5)`, current residual `(0.75, 0.75)`: num = absmean(0.25, 0.25) = 0.25, den = 0.5, diff = 0.5 exactly; threshold=0.5 sits exactly at the boundary (cache, strictly >); adding 2⁻⁴⁰ to the residual flips to compute. Residual `(0.15, 0.05)` against prev `(0.2, 0.1)`: difference `(−0.05, −0.05)`, diff after absmean = 0.05/0.15 = 1/3 → compute (an abs-less implementation gets −1/3 and wrongly caches).

Data scale: tensors `(1, 2)`–`(2, 2)` float64, threshold ∈ [0, 1].
