# KV Cache: xKV Cross-Layer Shared Singular Vectors

## Problem

Keys of neighbouring layers are strongly correlated, so xKV (2503.18893) moves low-rank compression from a single layer to a **group of layers**: concatenate the K of all layers in the group along the channel axis, take **one** SVD, share the resulting left singular vectors U across the whole group, and keep only each layer's own coefficient slice \(S\cdot V\). `layer_group_size` defaults to 1 (which degenerates to per-layer SVD); xKV uses 2 or 4. The rank defaults are 256 for K and 768 for V.

What matters is the **orientation** of the SVD. Upstream first transposes a layer's key from `[bsz, H, S, D]` to `[bsz, S, H*D]`, then `torch.cat(..., dim=2)` inside the group yields `[bsz, S, gs*H*D]`, and that matrix goes into the SVD:

- rows are **positions**, columns are the **concatenated channels**, flattened as "layer → head → dim";
- so U is the position-side factor `[S, rank]`, identical for every layer in the group — that is where "shared" comes from;
- each layer's `S·V` is just the column slice of `[rank, gs*H*D]` that belongs to it, reshaped to `[H, rank, D]`;
- reconstruction is `U @ SV_l`, one small matmul per layer, without rebuilding the full group width.

The rank truncates the **concatenated** matrix: `r = min(rank_k, S, gs*H*D)`. It may exceed a single layer's channel count `H*D` — widening the group and doubling the rank together is exactly how xKV compares configurations at equal compression.

Three details that are easy to get wrong:

1. the concatenation order is "layer first, then head-major inside the layer", not head-interleaved across layers;
2. the layer count must be divisible by `layer_group_size`: upstream only triggers the SVD when `layer_idx % gs == gs-1`, so a trailing partial group never gets merged (`U_k` is allocated with `num_layers // gs` rows) — layers are silently dropped;
3. upstream uses the randomised `torch.svd_lowrank`, whose result is not reproducible (the source comment says "Have deterministic issue but faster"). This problem pins the exact `torch.linalg.svd` plus a sign convention.

## Task

Implement `solve(k, layer_group_size=2, rank_k=256)`:

- `k`: `(num_layers, S, H, D)` float32/float64, the per-layer K (batch is always 1); all four dims must be >= 1;
- `layer_group_size`, `rank_k`: positive integers (bool does not count). A layer count that is not divisible raises `ValueError`.
- number of groups \(G=\texttt{num\_layers}/\texttt{gs}\), effective rank \(r=\min(\texttt{rank\_k}, S, \texttt{gs}\cdot H\cdot D)\);
- all arithmetic in float64, output in the input dtype; `k` must not be mutated.

**Sign convention (deterministic)**: for every column of U in every group, if the entry with the largest absolute value is negative, negate the whole column and the matching row of `S·V`; ties in absolute value are decided by the **smallest row index**. This leaves `u @ coef` unchanged while making U itself bit-reproducible.

Return a 4-key dict:

- `u`: `(G, S, r)`, the shared position-side singular vectors, one set per group;
- `coef`: `(num_layers, H, r, D)`, layer l's `S·V` slice `[l*H*D, (l+1)*H*D)` rearranged to `[H, r, D]`;
- `recon`: `(num_layers, S, H, D)`, equal to `u[l//gs] @ coef[l]`;
- `err`: `(G,)`, the per-group `‖A - U_r·S_r·V_r‖_F`.

## Example

```
layer_group_size = 2, rank_k = 1        # L=2 -> a single group
k[0] = [[2, 0], [0, 0]]                 # S=2, H=1, D=2
k[1] = [[0, 0], [0, 1]]

A = [k[0] | k[1]] = [[2, 0, 0, 0],      # (S=2, gs*H*D=4), column order [layer][head][dim]
                     [0, 0, 0, 1]]
sigma = [2, 1]                          # the two layers occupy orthogonal directions
r = min(1, 2, 4) = 1                    # keep only the sigma=2 column
u_r = [[1], [0]]        SV_r = [[2, 0, 0, 0]]
coef[0] = [[2, 0]]                      # (H=1, r=1, D=2): the first H*D entries
coef[1] = [[0, 0]]                      # the last H*D entries: this layer got no rank
recon  = [[[2, 0], [0, 0]], [[0, 0], [0, 0]]]
err    = ||A - u_r @ SV_r||_F = sigma_2 = 1.0
```

The rank budget belongs to the **group**: the direction living in `k[1]` has the smaller singular value, so the same `rank_k=1` squeezes it out. Raise `rank_k` to 2 and `err` becomes zero with `coef[1] = [[0, 0], [0, 1]]` — there is no such thing as a per-layer `rank_k`.

## Constraints

- **The rank truncates the concatenated width**: `r` is capped by `min(S, gs*H*D)` and may exceed `H*D`. A per-layer SVD can never obtain that extra rank.
- **U is shared, `S·V` is not**: there are only `G` copies of `u` but `num_layers` copies of `coef`. The tests pin this with "two identical layers in a group" (their coefficient blocks match element-wise) and with `S=1` (then `u=[1]` and `coef` must equal the original K element-wise, which fixes the column order).
- **The residual is the Eckart–Young tail**: under exact truncated SVD, `err = sqrt(Σ_{j>r} σ_j²)`, non-increasing in `rank_k`, and zero once the rank covers the matrix. The same repo uses this quantity as its yardstick in `test_svd_reconstruction_accuracy.py`.
- **Signs do not decide grading, but must be pinned**: the tests only compare `recon`/`err` and check the column-sign convention on `u`, never numpy's raw U — rotations inside an equal-singular-value block are a gauge, and a sign flip moves U and `S·V` together.
- **Do not SVD all layers at once**: groups must be independent; touching group 0 must not move group 1's `u`.
- **Empty sequence, non-4-D input, integer dtype, non-int `layer_group_size`** all raise `ValueError`.
