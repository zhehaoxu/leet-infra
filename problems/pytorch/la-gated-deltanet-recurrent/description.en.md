# Gated DeltaNet Recurrence Ledger: Gated Decay and Delta-Rule State Update

## Problem

Gated DeltaNet rewrites linear attention's state update into a "gated delta rule": the state matrix decays by `exp(g_t)` each step, then `β_t` writes "v minus the state's response to k" as an outer product. This is the core recurrence of 2412.06464; fla-org's `naive_recurrent_gated_delta_rule` is its per-token reference implementation. This problem asks you to transcribe the recurrence line by line and make every knob (`g`/`β`/`scale`/initial state) verifiable.

## Rules

Implement `solve(q, k, v, beta, g, scale=None, initial_state=None)` (inputs `[B,T,H,·]`), returning:

| Field | Shape | Meaning |
|---|---|---|
| `o` | `[B,T,H,V]` | per-step outputs (float32) |
| `final_state` | `[B,H,K,V]` | final state |

The recurrence (after `transpose(1,2)` and float32; `scale` defaults to `1/sqrt(K)` and **multiplies q first**):

```
h = initial_state (defaults to 0)
for i in range(T):
    h = h * exp(g_i)                                # gated decay (data-dependent, per head)
    v' = (v_i − h·k_i) * β_i                        # delta correction
    h = h + k_i ⊗ v'                                # outer-product write
    o_i = (q_i * scale) · h
```

## Example (K=V=1 scalar hand account)

```
h0=4, k=1, v=3, β=0.5, g=-0.1, scale=1
h ← 4·e^(-0.1) = 3.6398
v' ← (3 − 3.6398)·0.5 = −0.3199
h ← 3.6398 − 0.3199 = 3.3199
o ← q·h = 2·3.3199 = 6.6398
```

## Constraints

- `g ≤ 0` (log decay); `T ≤ 6`, `K,V ≤ 4`, the oracle is a pure-Python four-fold loop
- The decay must act **before** the delta correction (with `g→−∞` the state zeroes out and the correction reduces to `β·v`)
