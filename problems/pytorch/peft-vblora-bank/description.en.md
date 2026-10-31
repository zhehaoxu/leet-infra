# VB-LoRA: Vector-Bank Selection Adapter

## Problem Statement

VB-LoRA replaces the two LoRA matrices `A` and `B` entirely: the whole model shares a single **vector bank** (a few thousand to tens of thousands of rows of `vector_length` numbers), and each trainable parameter stores no vector at all — only a set of **scores (logits) over the bank rows**. At forward time the `topk` highest-scoring rows are picked, averaged with softmax weights, and `A` and `B` are assembled on the fly. That shrinks one adapter layer to a few thousand floats while the shared bank supplies the capacity.

Given the base weight `W0`, an optional bias `b0`, the `vector_bank`, the two logit tensors (`logits_A`, `logits_B`) and `topk`, return the assembled `A`, `B`, the delta `delta = B @ A` and the forward result `y`; the four must be mutually consistent.

## Rules

Let `r` be the rank, `V = num_vectors` and `vl = vector_length`. The last dimension of `x` is `d_in`; `W0` is `(d_out, d_in)`. The second-to-last dimension of each logit tensor is the number of **tiles** obtained by cutting input/output into pieces of length `vl`:

| tensor | shape | meaning |
| --- | --- | --- |
| `vector_bank` | `(V, vl)` | one bank shared globally |
| `logits_A` | `(r, d_in // vl, V)` | each (rank, input tile) scores all `V` bank rows |
| `logits_B` | `(d_out // vl, r, V)` | each (output tile, rank) scores all `V` bank rows |

1. **Selection per logit vector** (the last dimension of length `V` is one score vector):

   ```
   idx    = the topk largest scores of the V logits (ties: smaller index first)
   w      = softmax(logits[idx])           # normalized over the selected topk only
   vector = sum_t w[t] * bank[idx[t]]      # weighted sum of bank rows, length vl
   ```

   The crucial part is that **softmax is normalized over the `topk` selected scores only**; the discarded scores take no part at all. Running softmax over the whole length-`V` logit vector gives completely different weights.

2. **Assemble `A` (shape `(r, d_in)`)**: the selection result for `logits_A` has shape `(r, d_in//vl, vl)` and the last two dimensions are flattened in **tile-major order**,

   ```
   A[:, t*vl:(t+1)*vl] = selection(logits_A[:, t, :])
   ```

3. **Assemble `B` (shape `(d_out, r)`)**: the selection result for `logits_B` has shape `(d_out//vl, r, vl)`; **transpose to `(out_tile, vl, r)` first, then flatten the leading two dimensions**,

   ```
   B[o*vl + s, j] = selection(logits_B[o, j, :])[s]
   ```

   So the row index of `B` is the component `s` inside the output tile while the column index is the rank `j`; skipping this transpose scrambles the `vl` rows against the `r` columns within each output tile.

4. **Delta and forward**:

   ```
   delta = B @ A                # (d_out, d_in), no alpha/r style extra scaling
   y     = x @ W0.T + b0 + x @ delta.T
   ```

   When `b0` is `None` no bias is added. The second term equals multiplying by `A` first and then by `B` (`((x @ Aᵀ) @ Bᵀ)`), which is exactly the unmerged path.

5. **Precision**: do the intermediate math in float64 and return `y`, `A`, `B`, `delta` as float32; `x` may carry any number of leading dimensions.

Return `(y, A, B, delta)`. Calling an off-the-shelf PEFT library is forbidden.

## Example

`d_in = d_out = 2`, `vl = 2` (so both sides have a single tile), `V = 4`, `r = 2`, `topk = 2`, `W0 = I`, `x = [[1, 2]]`:

```
bank     = [[1, 0], [0, 1], [2, 2], [-1, 0.5]]
logits_A = [[[0, 1, 0.5, -1]],
            [[0, 0, 3, 1]]]
logits_B = [[[1, 0, 0.5, 2],
             [1, 1, 1, -2]]]
```

The two rows of `logits_A` are the input tiles of rank 0 and rank 1; the two rows of `logits_B` are rank 0 and rank 1 of output tile 0.

- `logits_A[0]` selects indices 1 (score 1.0) and 2 (score 0.5); `softmax([1, 0.5]) = [0.622459, 0.377541]`, giving `0.622459*[0,1] + 0.377541*[2,2] = [0.755081, 1.377541]`.
- `logits_A[1]` selects indices 2 (score 3.0) and 3 (score 1.0); `softmax([3, 1]) = [0.880797, 0.119203]`, giving `[1.642391, 1.821196]`.
- `logits_B[0, 0]` selects indices 3 (score 2.0) and 0 (score 1.0); `softmax([2, 1]) = [0.731059, 0.268941]`, giving the vector `[-0.462117, 0.365529]`.
- For `logits_B[0, 1]` the maximum `1.0` ties across indices 0/1/2, and `topk = 2` takes 0 and 1 in ascending index order with weights 0.5 each, giving `[0.5, 0.5]`.

```
A     = [[0.755081, 1.377541],
         [1.642391, 1.821196]]
B     = [[-0.462117, 0.500000],
         [ 0.365529, 0.500000]]
delta = [[0.472260, 0.274013],
         [1.097200, 1.414129]]
y     = [[2.020285, 5.925458]]
```

Note that column 0 of `B` is `[-0.462117, 0.365529]` — the two components of the rank-0 vector — and column 1 is `[0.5, 0.5]`: components run down the rows and ranks across the columns, which is what the transpose produces.

## Constraints

- `d_in` and `d_out` must both be divisible by `vl`; `vl`, `V` and `r` are positive integers;
- `0 < topk <= V`;
- `logits_A` and `logits_B` must match the table exactly (`r` and `V` have to agree with `vector_bank`);
- the last dimension of `x` must equal `d_in`; if `b0` is given its length must equal `d_out`;
- every tensor must be floating point and finite (`NaN`/`Inf` are invalid);
- violating any of the above raises `ValueError`.
