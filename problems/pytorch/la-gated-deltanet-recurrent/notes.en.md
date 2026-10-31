# Notes: Gated DeltaNet Recurrence Ledger

## Provenance

`fla-org/flash-linear-attention` — `fla/ops/gated_delta_rule/naive.py::naive_recurrent_gated_delta_rule` (fetched line-by-line via the GitHub contents API on 2026-10-31). Paper source: Gated DeltaNet (2412.06464), `S_t = α_t S_{t-1}(I − β_t k_t k_tᵀ) + β_t k_t v_tᵀ` with `α_t = exp(g_t)`. The fla reference rewrites it as the equivalent "decay → delta correction → outer-product write" three-step, numerically identical to the paper form.

## Key points and traps

1. **Decay-before-correction order**: `h *= exp(g)` must precede `v' = v − h·k` — the correction reads the decayed state. Moving the decay after the write is caught by `gating_order` (at `g=−50` the state ≈ 0 and the correction reduces to `β·v`) and the `delta_rule_corrects` scalar account.
2. **`β` is a per-head scalar**: `v' = (v − h·k)·β` and "multiply β into k" (`h += (βk)⊗(v−hk)`) are mathematically equivalent (both outer-product factors scale identically) — mutation testing records `beta-on-k` as an **observationally equivalent mutation**. Dropping β or mis-axing it is not equivalent.
3. **`scale` multiplies q exactly once**: default `1/sqrt(K)`; the mutations "no q scale", "default as 1.0", and "divide at the output" are all separated by default_scale / random comparison — `scale-after` (drop the multiply) vs `o-no-scale` (divide at read) also differ in rounding (`q*scale` then inner product ≠ inner product then divide), and the difference clears the 1e-4 tolerance.
4. **Initial state composed with gating**: at `g=0, β=0`, `o_t = q_t·scale·h_0` stays constant and the final state equals the initial state — the cleanest mechanism anchor; a non-float32 `initial_state` is caught by the dtype ledger.
5. **Output layout**: the recurrence runs in `[B,H,T,·]` and `o` must transpose back to `[B,T,H,V]` — the oracle's first draft transposed in the wrong order (V/H swapped) and only V>1 random cases exposed it, proving that small-dimension tests must cover V≠H.

## Mutation testing (8 effective caught + 1 equivalent)

drop decay, move decay after the correction, drop β, β onto k (equivalent), drop the delta correction, no q scale, default scale as 1.0, drop the initial state, divide by scale at the output.

## Complexity

O(B·H·T·K·V) recurrence, scale ≤ 2×3×6×4×4, milliseconds.
