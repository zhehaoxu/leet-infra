# Mamba Selective Scan: ZOH Discrete Recurrence and Gated-Skip Ledger

## Problem

Mamba's (S6) core move is making the state-space A/B/C input-dependent (selective); after discretization each step is a single elementwise linear recurrence. The official `selective_scan_ref` is the authoritative reference for this recurrence. This problem asks you to transcribe its arithmetic ledger item by item: the ZOH exponential, the `ΔB·u` aggregation, static-vs-variant branches of B/C, the D skip, the z gate, and the `delta_bias + softplus` pre-processing.

## Rules

Implement `solve(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False)`, transcribing `selective_scan_ref` (real A), returning:

| Field | Shape | Meaning |
|---|---|---|
| `out` | `[B, D, L]` | same dtype as `u` |
| `last_state` | `[B, D, N]` | final state `x` |

The recurrence (compute after `.float()`, complex A raises `ValueError`):

```
deltaA   = exp(delta ⊗ A)                      # 'bdl,dn->bdln'
deltaB_u = delta ⊗ B ⊗ u                       # static B 'bdl,dn,bdl->bdln'
                                               # variant B 'bdl,bnl,bdl->bdln'
x = 0
for i in range(L):
    x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
    y_i = x·C  (static 'bdn,dn->bd' / variant 'bdn,bn->bd' with C[:, :, i])
out = y；out += u * D[:, None] if D given
out = out * silu(z) if z given
```

`delta` pre-processing order: add `delta_bias` first, then `softplus`. B and C each independently support static `[D,N]` and variant `[B,N,L]` branches.

## Example

```
D=1, N=1, L=2, A=-1, B=C=1, delta=[0, ln2], u=[1,2]
step 1: deltaA=1, ΔBu=0 → x=0, y=0
step 2: deltaA=exp(-ln2)=1/2, ΔBu=ln2·1·2 → x=2ln2, y=2ln2
out=[0, 2ln2], last_state=2ln2
```

## Constraints

- `A` negative real (stable `exp`); scale `D/N/L ≤ 3×3×6`, the oracle is a pure-Python five-fold loop
- The chunked-scan boundary carry (split L, pass the final state across segments) is mathematically identical to the full recurrence
