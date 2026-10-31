# Activation Steering: Vector Injection and Direction Ledger

## Problem

Activation steering (representation engineering / steering vectors, source: `nrimsky/steering-vectors`) adds a direction vector with a coefficient into the residual stream at chosen layers during inference to pull generation toward a target behavior. This problem turns injection into a deterministic ledger: given per-layer hidden states `hiddens [L, T, D]`, a steering vector `s [D]`, an injection sequence `(alphas[k], layer_indices[k])` and optional pre-normalization, return the steered hidden states plus the per-layer per-token directional account (cosines before/after, projection coefficients, per-layer accumulated injection norm).

## Solution

```
s_eff = s/|s| (normalize=True) or s
for k = 0..K-1 in order: steered[layer_indices[k]] += alphas[k] * s_eff   # accumulate per layer
Ledger (all float64):
  layer_delta[l] = |Σ_{k: idx=l} α_k| · |s_eff|     # 0 for untouched layers
  cos_before[l][t] = cos(h, s)      # from the ORIGINAL hiddens, independent of order
  cos_after[l][t]  = cos(h', s)
  proj_coef[l][t]  = (h·s)/(s·s)    # the α that makes h' ⊥ s; injecting −proj_coef removes the s component
```

Key points: repeated injections at the same layer **accumulate** (never overwrite); `steered` must be a copy (never mutate the input in place — for float64 inputs `.double()` does not copy); `cos_before` must come from the original hidden states; the `proj_coef` denominator is `|s|²`, not `|s|`.

## Data scale

- 1 ≤ L ≤ 8, 1 ≤ T ≤ 8, 1 ≤ D ≤ 32, 1 ≤ K ≤ 8.
- Shape mismatches, zero `s`, out-of-range/non-integer layer indices, non-finite/bool alphas, K=0 → `ValueError`.
