# Notes: Mamba Selective Scan Ledger

## Provenance

`state-spaces/mamba` — `mamba_ssm/ops/selective_scan_interface.py::selective_scan_ref` (fetched line-by-line via the GitHub contents API on 2026-10-31). This is the authoritative reference implementation of the S6 layer from the Mamba paper (2312.00752): `ΔA = exp(Δ·A)` (first-order ZOH), `ΔB_u = Δ·B·u`, `x_t = ΔA x_{t-1} + ΔB_u`, `y_t = C_t·x_t`. The complex branch (`y.real * 2`) and the G-group repeat (`repeat(B, "B G N L -> ...")`) are out of scope — reasons in the notes (weak judgeable signal for complex; the G group adds only a repeat wiring step).

## Key points and traps

1. **`delta` pre-processing order**: `+ delta_bias` first, then `softplus` — the reversed order is caught by `check_delta_bias_softplus`'s equivalent-rearrangement assertion (softplus(b)+Δ ≠ softplus(b+Δ)).
2. **Four B/C combinations**: static/variant decided independently (`B.dim() >= 3`). The mutation "variant B with the static einsum" replaces `B[b,n,l]` with `B[d,n]` — when dim==bN and N==L the shapes happen not to break but the values are wrong; random comparison catches it.
3. **`last_state` is the final `x`** (step L-1), not an alias of the output; the "subtract the last injection" mutation is caught directly by the hand anchor (`2ln2` becomes `2ln2 − ΔBu`).
4. **The dtype ledger**: upstream `u.float()` pulls everything to float32 for compute and `.to(dtype_in)` at the end — a float64 input yields a float64 output computed **at float32 precision** (upstream-consistent); a missing `A.float()` makes the einsum raise on dtype directly (tested).
5. **Chunked-carry identity**: splitting L and feeding segment 1's final state as segment 2's initial state matches the full recurrence bitwise — the mathematical basis for Mamba's chunked scan correctness; the test verifies via an oracle-side segmented run. The solution takes no initial-state argument (narrow interface); the segmented equivalence lives in the oracle.
6. **z gating after the D skip**: `out = (y + u*D) * silu(z)` — skip first, gate second; an order-swap mutation is caught by the z case.

## Mutation testing (9/9 caught)

missing exp, no state update (drop injection), variant B with static einsum, D skip removed, z gate without silu, softplus/bias swapped, last_state not the final state, variant C with static einsum, dtype cast-back removed.

## Complexity

O(B·D·N·L) recurrence, scale ≤ 2×3×3×6, milliseconds.
