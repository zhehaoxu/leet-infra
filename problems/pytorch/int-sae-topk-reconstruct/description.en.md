# SAE Top-k Encoding Ledger: Sparse Activations, Reconstruction and L0 Sparsity

## Problem

A sparse autoencoder (SAE) decomposes model hidden states into high-dimensional sparse features: the encoder keeps the k largest pre-activations and zeroes the rest; the decoder reconstructs the hidden state from these sparse activations. The boundary between TopK SAE (the SAELens architecture) and a plain ReLU SAE is **no ReLU** — negative pre-acts are selected as-is. This problem asks you to reconcile the encode/decode ledger and the sparsity metrics item by item.

## Rules

Implement `solve(x, W_enc, b_enc, W_dec, b_dec, k)`, returning:

| Field | Shape | Meaning |
|---|---|---|
| `feature_acts` | `[B, F]` | top-k sparse activations (others strictly 0, negatives kept) |
| `recon` | `[B, D_in]` | `acts @ W_dec + b_dec` |
| `recon_error` | `[B, D_in]` | `x − recon` |
| `l0` | `[B]` | count of nonzero activations |
| `mse` | `[B]` | row-wise mean square of `recon_error` (over D_in) |
| `max_feat` | `[B]` | index of the largest-|activation| (ties → smaller index) |

Steps (in float):

```
pre_acts = (x − b_dec) @ W_enc.T + b_enc
per row keep the k largest values (ties → smaller index), zero the rest
recon = acts @ W_dec + b_dec
```

Raise `ValueError` unless `1 ≤ k ≤ F`.

## Example

```
x=[1,2]，W_enc rows e0/e1/(e0+e1)/2，b_enc=[0,0,1]，W_dec rows e0/e1 + a zero row，b_dec=0，k=2
pre = [1, 2, 2.5] → keep feat2(2.5), feat1(2.0)
recon = [0, 2]，mse = (1² + 0²)/2 = 0.5，max_feat = 2
```

## Constraints

- `F ≤ 16`, `D_in ≤ 8` small scale, pure-Python two-fold oracle
- At `k = F` the activations equal the full pre_acts (no zeroing)
