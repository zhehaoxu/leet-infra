# RoAD: Grouped 2-D Rotation Adapter

## Problem Statement

RoAD (Rotation Adaptation) does not add a low-rank term to the weight; it **rotates the base layer's output in 2-D pairs**, with its own angle and scale for every pair inside every group. It never stores an `out_features × out_features` rotation matrix — only angles `theta` and scales `alpha` — and at inference time uses the elementwise form `y = x * first_col + rotate_half(x) * second_col` instead of a matmul.

Given angles, scales, a variant, a group size, and an input tensor, produce the **explicit rotation matrix `R`** and the **rotated output `y`**, and make sure they agree (`y = x @ Rᵀ`).

## Rules

Let `out_features` be the output width, `group_size` a positive even number, and `half = group_size // 2`. Split the output into `out_features / group_size` groups; **inside each group, element `j` pairs with element `j + half`** (not with its neighbor).

1. **Parameter lengths** (`theta` and `alpha` are equal-length):

   | variant | length | meaning |
   | --- | --- | --- |
   | `road_1` | `out_features / 2` | each group has `group_size/2` parameters, shared by a pair |
   | `road_2` | `out_features` | one parameter per position |
   | `road_4` | `out_features * 2` | two parameter sets per pair: the first half feeds the `cos` term, the second the `sin` term |

2. **Per-position column coefficients** `first_col[k]`, `second_col[k]`:

   - `road_1`: reshape `theta` and `alpha` to `(n_group, half)`, then `repeat_interleave(2)` on the second dim and flatten; then `first_col = alpha * cos(theta)`, `second_col = alpha * sin(theta)` elementwise.
   - `road_2`: keep the original order and use the same formulas.
   - `road_4`: reshape `theta` and `alpha` to `(n_group, 2, group_size)`; `first_col = alpha[:, 0, :] * cos(theta[:, 0, :])`, `second_col = alpha[:, 1, :] * sin(theta[:, 1, :])`, each flattened by group.

3. **Rotation matrix**:

   ```
   R = diag(first_col) + contribution of the second column
   ```

   Building the second-column contribution: reshape `second_col` to `(n_group, 2, half)`, swap the two halves, flatten, take the diagonal; reshape that to `(n_group, 2, half, out_features)`, swap the halves again, negate the first half, and reshape to `out_features × out_features`.

   The equivalent block view (for each pair `a = g*group_size + j`, `b = a + half`):

   ```
   R[a, a] = first_col[a]
   R[b, b] = first_col[b]
   R[a, b] = -second_col[a]
   R[b, a] =  second_col[b]
   ```

4. **Elementwise output**: split `x` into halves `x1`, `x2` within each group,

   ```
   rotate_half = stack(-x2, x1)         # same shape as x
   y = x * first_col + rotate_half * second_col
   ```

   which must equal `x @ Rᵀ`; with `alpha` all ones, `R` is orthogonal and preserves the 2-norm.

The last dim of `x` must be `out_features`; any leading dims are allowed. Use float64 intermediate math and return float32.

Return `(R, y)`: `R` is `(out_features, out_features)`, `y` matches `x`'s shape.

## Example

`theta = [0.1, 0.2, 0.3, 0.4]`, `alpha = [1, 1, 1, 1]`, `variant = "road_1"`, `group_size = 4`, `out_features = 8`, `x = [[0, 1, 2, 3, 4, 5, 6, 7]]`

The first group's four positions share two angles (`0.1` for positions 0/2, `0.2` for 1/3):

```
R[:4, :4] = [[ 0.9950,  0.0,    -0.0998,  0.0   ],
             [ 0.0,     0.9801,  0.0,    -0.1987],
             [ 0.0998,  0.0,     0.9950,  0.0   ],
             [ 0.0,     0.1987,  0.0,     0.9801]]
```

The second group (angles `0.3`, `0.4`) lives in `R[4:, 4:]`, and cross-group entries stay 0.

```
y = [[-0.1997, 0.3841, 1.9900, 3.1389, 2.0482, 1.8794, 6.9141, 8.3945]]
```

## Constraints

- `group_size` must be a positive even number and `out_features` must be divisible by it;
- `variant` must be one of `road_1`, `road_2`, `road_4`;
- `theta` and `alpha` must be 1-D with the lengths above;
- the last dim of `x` must equal `out_features`;
- raise `ValueError` when any of these fails.
