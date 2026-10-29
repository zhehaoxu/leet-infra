# HRA: Householder Reflection Adaptation with a Compact WY Merge

## Problem Statement

HRA (Householder Reflection Adaptation) leaves the base weight alone and multiplies the input side by an orthogonal matrix `H`: `H` is the product of `r` Householder reflections, each determined by a trainable vector column `u_k`. At merge time `H` is folded into the base weight (`W0 @ H`), so inference reduces to a plain linear layer.

The subtle part: multiplying `r` reflections one after another produces cross terms, so `H` cannot simply be written as `I - V Vᵀ`. HRA compresses it with the compact WY representation `H = I - V T Vᵀ`, where `T` is an `r×r` upper-triangular matrix.

Given the reflection vectors `u`, the base weight `W0`, whether to run Gram-Schmidt first, and a forward/reverse direction, produce `V`, `T`, `H`, and the merged weight.

## Rules

Let `u` be `(in_features, r)`, `W0` be `(out_features, in_features)`, and `rank = r`. Do all intermediate math in float64 and return float32.

1. **Orthonormalized branch** (`apply_gs=True`): take the reduced QR decomposition of `u` and set `V = Q` (`in_features × r`), then

   ```
   T = 2 * I_r
   ```

   Here `H = I - V T Vᵀ = I - 2 Q Qᵀ`: identity on the orthogonal complement of span(u), negation on span(u).

2. **Plain branch** (`apply_gs=False`): each column of `V` is the unit-normalized version of the matching `u` column,

   ```
   v_k = u_k / ||u_k||
   T_inv = 0.5 * I_r + triu(Vᵀ V, diagonal=1)
   T     = solve_triangular(T_inv, I_r, upper=True)
   ```

   `triu(·, diagonal=1)` keeps the strictly-upper triangle only. The resulting `H = I - V T Vᵀ` equals the sequential product `P_0 @ P_1 @ … @ P_{r-1}` of reflectors (`P_k = I - 2 v_k v_kᵀ`); `T` restores the cross terms.

3. **Reverse** (`reverse=True`): replace `T` with `Tᵀ`, yielding `Hᵀ`. Used by unmerge.

4. **Merge**: `W_merged = W0 @ H`, equivalent to applying `x @ Hᵀ` to the input before the base linear layer.

Return `(V, T, H, W_merged)` with shapes `(in_features, r)`, `(r, r)`, `(in_features, in_features)`, and `(out_features, in_features)`.

## Example

`u = [[1, 2], [0, 1], [0, 0]]`, `W0 = [[1, 2, 3], [4, 5, 6]]`, `apply_gs = False`

The two columns are not orthogonal (their normalized dot product is nonzero), so `T` has a nonzero upper-right entry:

```
V = [[1.0, 0.8944272],
     [0.0, 0.4472136],
     [0.0, 0.0      ]]
T = [[2.0, -3.5777087],
     [0.0,  2.0      ]]
H = [[ 0.6, 0.8, 0.0],
     [-0.8, 0.6, 0.0],
     [ 0.0, 0.0, 1.0]]
W_merged = [[-1.0,  2.0, 3.0],
            [-1.6,  6.2, 6.0]]
```

`H` is a product of two reflections, which in 3-D acts as a rotation on the first two coordinates (`det H = 1`) and leaves the third unchanged.

With the same `u` and `apply_gs = True`, `V` is the QR orthonormal basis of `u` (here just `[[1,0],[0,1],[0,0]]`) and `T = 2I`:

```
H = [[-1.0,  0.0, 0.0],
     [ 0.0, -1.0, 0.0],
     [ 0.0,  0.0, 1.0]]
W_merged = [[-1.0, -2.0, 3.0],
            [-4.0, -5.0, 6.0]]
```

With `reverse=True`, `H` is transposed: the `0.8` becomes `-0.8` and the `-0.8` becomes `0.8`.

## Constraints

- `u` must be 2-D, `r >= 1`, and no column may be the zero vector;
- `W0.shape[1] == u.shape[0]`;
- `apply_gs=True` requires `r <= in_features`;
- raise `ValueError` when any of the above is violated.
