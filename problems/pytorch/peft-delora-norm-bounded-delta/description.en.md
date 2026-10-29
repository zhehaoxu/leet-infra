# DeLoRA: A Bounded Low-Rank Update Normalized by Row Norms

## Problem Statement

A plain LoRA update `B A` is unbounded, so a large training step can push the base weight far away. DeLoRA gives each rank-1 component its own scale so that every component has norm exactly `lam / r`; the whole update then has Frobenius norm at most `lam`. It additionally multiplies by `w_norm`, the per-column norm of the base weight, so the update adapts to input-channel scale.

Given the base weight, low-rank factors, the bound parameter, and the column norms, produce the per-component scales, the update matrix, and the merged weight.

## Rules

Let `rank = r` (taken from `A`'s row count), `An_i = max(||A_i||_2, 1e-4)` (norm of row `i` of `A`), and `Bn_i = max(||B^i||_2, 1e-4)` (norm of column `i` of `B`).

1. **Per-component scale**:

   ```
   scaling_i = lam / r / (An_i * Bn_i)
   ```

   The `1e-4` floor guards zero rows/columns, which would otherwise send the scale to infinity.

2. **Update matrix**:

   ```
   delta = B @ diag(scaling) @ A
   delta = delta * w_norm          # column-wise (input-channel) scaling
   ```

3. **Merge**: `merged = W0 + delta`.

Key point: the scale depends **only** on `A`'s row norms and `B`'s column norms, never on data; `w_norm` is the base weight's column norms frozen at training start and is not updated. When the rows of `A` and columns of `B` are unit vectors, the rank-1 components are mutually orthogonal and `delta` has Frobenius norm exactly `lam / sqrt(r)`.

Return `(scaling, delta, merged)`: `scaling` has shape `(r,)`, `delta` and `merged` have shape `(out_features, in_features)`, all float32.

## Example

`W0 = [[1, 2, 3], [4, 5, 6]]`, `A = [[1, 0, 0], [0, 2, 0]]`, `B = [[1, 0], [0, 3]]`, `lam = 1.5`, `w_norm = [1, 2, 1]`

```
An = [1, 2]        Bn = [1, 3]
scaling = [1.5/2/(1*1), 1.5/2/(2*3)] = [0.75, 0.125]

delta = B @ diag([0.75, 0.125]) @ A = [[0.75, 0, 0],
                                       [0,  0.75, 0]]
delta * w_norm (column-wise, w_norm = [1, 2, 1]) = [[0.75, 0, 0],
                                                    [0,  1.5, 0]]

merged = [[1.75, 2.0, 3.0],
          [4.0,  6.5, 6.0]]
```

If `A`'s first row were all zeros, `An_0` is clamped to `1e-4` and `scaling_0 = lam/2/(1e-4 * 1)`; with `lam=1.0` and `Bn_0=1` that is `5000`.

## Constraints

- `A` and `B` must be 2-D and `rank >= 1`;
- `B.shape[1] == A.shape[0]`;
- `W0.shape == (B.shape[0], A.shape[1])`;
- `w_norm` is 1-D with length `A.shape[1]`;
- `lam >= 0`;
- raise `ValueError` when any of these fails.
