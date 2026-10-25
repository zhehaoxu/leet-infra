# Walkthrough: Exponential Moving Average (EMA) Weights with Warmup

## Source and verification
- **EMA shadow weights**: the idea appears in Mean Teacher (Tarvainen & Valpola, arXiv:1703.01780) — an exponential moving average of the weights over time yields a more stable target model; diffusion models (DDPM/EDM and friends) and Model Soup evaluation averaging use it widely too.
- **Warmup schedule \(d_n=\min\!\left(decay,\frac{1+n}{10+n}\right)\)**: verified as a verbatim implementation of llava's `model/ema.py` (inherited from MAE / `timm`'s `EMAHelper`) — the original comment reads exactly `decay = min(decay, (1 + num_updates) / (10 + num_updates))`. Meaning: while the number of updates is tiny, keep \(d\) small so the shadow catches up with the current weights quickly instead of being "nailed" to the random initialization; only for large \(n\) does \(d\) cap out at `decay`.

## Core idea
\[shadow \leftarrow d\cdot shadow+(1-d)\cdot param\]
The closer \(d\) is to 1, the "lazier" and smoother the shadow is; warmup makes \(d\) climb from a low value. This problem unifies the two paths — "warm up by update count" and "fixed decay" — into a single \(d_n\).

## Why it is easy to get wrong (pitfalls)
1. **The base of \(n\) and when it increments**: at construction \(n=0\) and \(shadow=p\) (untouched); **each `update()` first does \(n\leftarrow n+1\) and then computes \(d_n\)**. So the first `update()` uses \(n=1\) (\(d=2/11\)), not \(n=0\).
2. **The shadow is a clone, not a reference**: `shadow` must be a tensor separate from `param`; if it aliases the same memory, `update()` would rewrite `param` along with it, and `param` gets corrupted from \(n=1\) onward.
3. **In place, shadows only**: `update()` must not modify `params` (no `zero_grad`, no writing back into the parameters). It reads the **current values** of `params` — the judge mutates `params` in place between two `update()` calls.
4. **Warmup is capped**: \(\min(decay,\cdot)\), not plain \((1+n)/(10+n)\); the latter tends to 1 as \(n\to\infty\) and would exceed the given `decay`. With `warmup=False` it is always `decay`.
5. **All parameters share \(d_n\)**: one `update()` applies the same scalar \(d_n\) to every shadow, not a per-parameter counter.

## Comparison setup
The judge recomputes independently with `_oracle_ema` (feeding the whole parameter trajectory through a pure Python loop using the non-in-place form `d*n + (1-d)*p`) and compares against the reference solution with `allclose(atol=1e-5, rtol=1e-5)`. The example is fully deterministic (no randomness). Checks cover:
- `check_example`: with warmup, the concrete shadow values of a single parameter after 7 `update()` calls.
- `check_matches_oracle_multi`: several parameters, mixed shapes, `param` mutated step by step, shadows cross-checked against `_oracle_ema`.
- `check_no_warmup_fixed_decay`: `warmup=False`, `decay=0.5`, verifying \(d=0.5\) on every step.
- `check_decay_zero_tracks_exactly`: `warmup=False`, `decay=0.0` → the shadow equals the current `param` every time.
- `check_shadow_independent_and_params_untouched`: after `update()`, the `params` values are unchanged and `shadow` is not the same tensor as `param`.

## Complexity
Each `update()` is \(O(\text{numel})\); the cost is one extra set of shadow weights (roughly as large as the model itself in memory).
