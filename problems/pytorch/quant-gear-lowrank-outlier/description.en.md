# Quantization: GEAR Low-Rank Power-Iteration Error Compensation

## Problem

GEAR (arXiv 2403.05527) does not modify the quantizer itself; it decomposes the error the quantizer leaves behind one more time: a sparse term absorbs the outliers and a low-rank term absorbs the remaining quantization residual, so the KV cache is stored as "low-bit codes + a few outliers + a pair of low-rank factors". This problem implements the three stages of that pipeline in order.

1. **Outliers (the S of GEAR)**: merge `(B, H, L, D)` along the head and head_dim axes into token rows `(B, L, C=H*D)`; for each row take the `n_out` smallest and the `n_out` largest entries, replace them by **the original mean of the whole row** (the outliers themselves included) before quantizing, then write the **original values** of those positions back after quantization, so the residual at outlier positions is identically 0;
2. **Group-wise asymmetric quantization**: each token row is cut into consecutive `group_size` groups, scale \(=(mx-mn)/(2^{\text{bits}}-1)\), code \(=\mathrm{round}(\max((x-mn)/\text{scale},\,0))\), dequantized \(\hat x=\text{code}\cdot\text{scale}+mn\);
3. **Low-rank compensation (the L of GEAR)**: for the residual \(A\) of every \((b,h)\) (shape \(L\times D\)) run `loop` power-iteration rounds, with both QR steps happening **only in the last round**; at the end the left and right bases \(Q,P\) are orthonormal and \(P=A^{\top}Q\), so `low_rank` \(=Q P^{\top}=Q Q^{\top}A\) is the **orthogonal projection** of \(A\) onto \(\operatorname{span}(Q)\) — the compensation can only shrink the error, never grow it; `output` \(=\) `quantized` \(+\) `low_rank`. The row-wide budget, the outlier-inclusive mean and the \(2^{\text{bits}}-1\) level count are all graded conventions.

## Task

Implement `solve(kv, bits, group_size, sparsity, rank, loop, p_base, q_base)`:

- `kv`: float32/float64, `(n_batch, n_head, seq_len, head_dim)`, the K/V layout used by the upstream GEARLM;
- `bits`, `group_size`, `rank`, `loop`: positive integers; `group_size` must divide `n_head*head_dim`, and `rank <= min(seq_len, head_dim)`;
- `sparsity`: a real number in \([0,1)\), the outlier fraction of a whole row;
- `p_base`: `(n_batch, n_head, head_dim, rank)`, the initial right basis; `q_base`: `(n_batch, n_head, seq_len, rank)`, the initial left basis; both share `kv`'s dtype and are read-only.

All arithmetic inside is float64, return values use `kv`'s dtype, and all six keys have the same shape as `kv`:

- `outlier_mask`: bool, the positions selected as outliers;
- `sparse_filled`: the tensor after both ends of extremes were replaced by the row mean;
- `quantized`: the tensor after group quantization with the original extremes written back;
- `residual`: `kv - quantized`, exactly 0 at outlier positions;
- `low_rank`: the rank-`rank` power-iteration approximation of the residual;
- `output`: `quantized + low_rank`.

Raise `ValueError` on: `kv` not 4-D or not float32/float64; any of the four integer arguments not an int (`bool` does not count) or smaller than 1; `sparsity` being a `bool` or outside \([0,1)\); `group_size` not dividing `n_head*head_dim`; `rank` exceeding `min(seq_len, head_dim)`; a basis with the wrong shape or a dtype differing from `kv`.

## Example

```
kv[0, 0] = [[-4, 1, 2, 3], [-4, 1, 2, 3], [-4, 1, 2, 3]]      # head0
kv[0, 1] = [[ 4, 5, 6, 7], [ 4, 5, 6, 8], [ 4, 5, 6, 9]]      # head1
bits = 2   group_size = 4   sparsity = 0.25   rank = 1   loop = 1
p_base = ones(1, 2, 4, 1)      q_base = zeros(1, 2, 3, 1)

# 1) row-wide budget: after merging heads each row has 8 channels, 1 outlier at each end
n_out = int(int(1*2*3*4*0.25)/1/3/2) = int(6/6) = 1
row0 = [-4, 1, 2, 3 | 4, 5, 6, 7]   mean = 24/8 = 3           # the mean includes -4 and 7
row1 = [-4, 1, 2, 3 | 4, 5, 6, 8]   mean = 25/8 = 3.125
row2 = [-4, 1, 2, 3 | 4, 5, 6, 9]   mean = 26/8 = 3.25        # each row loses -4 and its own 7/8/9
sparse_filled[head0] = [[3, 1, 2, 3], [3.125, 1, 2, 3], [3.25, 1, 2, 3]]
sparse_filled[head1] = [[4, 5, 6, 3], [4, 5, 6, 3.125], [4, 5, 6, 3.25]]

# 2) group quantization (2**bits - 1 = 3 intervals), on the head1 group of row1, [4, 5, 6, 3.125]
mn = 25/8   mx = 6   scale = (6 - 25/8)/3 = 23/24
4     -> (4 - 25/8)/scale = 21/23 ≈ 0.9130 -> code 1 -> 25/8 + 23/24 = 49/12 ≈ 4.0833
5     -> (5 - 25/8)/scale = 45/23 ≈ 1.9565 -> code 2 -> 25/8 + 23/12 = 121/24 ≈ 5.0417
6     -> (6 - 25/8)/scale = 3             -> code 3 -> 6
3.125 -> 0                                -> code 0 -> 25/8, then the original outlier 8 is written back at the group end
quantized[head1, row1] = [49/12, 121/24, 6, 8]   residual[head1, row1] = [-1/12, -1/24, 0, 0]

# 3) low-rank compensation: the residual A of each (b, h) is a 3x4 matrix
residual[head1] = [[0, 0, 0, 0], [-1/12, -1/24, 0, 0], [-1/6, -1/12, 0, 0]]     # row 3 = 2x row 2, rank 1
residual[head0] = [[0, 0, -1/3, 0], [0, 0, 7/24, -1/8], [0, 0, 1/4, -1/4]]      # two independent columns, rank 2
qr(p_base).Q supplies a single direction (1, 1, 1, 1)/2, so a rank-1 projection cannot cover a rank-2 residual:
low_rank[head0] = [[0, 0, -23/60, 1/20], [0, 0, 23/120, -1/40], [0, 0, 0, 0]]
output[head1] == kv[head1] (restored element by element)    ||kv - output|| = 0.3873 < ||kv - quantized|| = 0.6166
```

In head1 the first row lands exactly on grid points (its scale is 1), so the residual survives only in the first two columns of rows 2 and 3, and those two rows are proportional — mathematically rank 1, which the rank-1 projection compensates fully and restores the input element by element; head0's residual occupies two independent columns, i.e. rank 2, so a rank-1 projection only takes away slightly more than half of it.

## Constraints

- **The budget is granted on the merged row and truncated twice**: `n_out = int(int(B*H*L*D*sparsity)/B/L/2)` — the inner cast truncates the element count first, and the divisor contains only `B`, `L` and `2` (never `H`). For the example shape, `sparsity` of 0.24/0.25/0.5 gives `n_out` of 0/1/2.
- **The replacement value is the original row mean, outliers included**: not a trimmed mean. The row `[-100, 0, 0, 0, 0, 0, 0, 100]` becomes all zeros after replacement; the two ends are written in two passes, and an index selected by both ends ends up with the same value either way.
- **Ties go to the smaller index (at both ends)**: the implementation uses a stable argsort, equivalent to sorting by `(value, index)` and taking the first `n_out`; upstream's `torch.topk` leaves tie order undefined.
- **The level count is `2**bits - 1` intervals**: codes live in `[0, 2**bits - 1]`. It is not `bits**2 - 1` — that one is an upstream typo, which coincides with the correct formula only for `bits ∈ {2, 4}`.
- **A constant group dequantizes to the group min**: when `scale == 0` no division happens (the denominator falls back to 1), the code is always 0 and the result is `mn`, never NaN.
- **`q_base` is free, `p_base` fixes the subspace**: for `loop >= 1` the `q_base` is entirely overwritten by `A @ P` in the first round, so any value leaves `low_rank` element-wise identical; `p_base` only matters through the subspace spanned by its QR orthonormalized columns in the last round, so negating or rescaling it does not change `low_rank`.
- **The compensation can only shrink the error**: `low_rank` is an orthogonal projection, hence `||kv - output|| <= ||kv - quantized||`, and each \((b,h)\) block of `low_rank` has rank at most `rank`.
- **Dtype and side effects**: everything is computed in float64; `outlier_mask` is always bool and the other five keys follow `kv`'s dtype; the three input tensors must not be modified in place, and a non-contiguous view must give the same result as a contiguous copy.
