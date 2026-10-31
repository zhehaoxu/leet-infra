# SHiRA Sparse Delta: COO Scatter Reconstruction with Duplicate Accumulation

## Problem

LoRA organizes trainable parameters into two dense low-rank matrices; SHiRA (Sparse High-Rank Adaptation, huggingface/peft `tuners/shira`) drops structure entirely: it picks $K$ **discrete coordinates** in the weight matrix and trains only those $K$ scalars. The parameter budget matches LoRA — $K = r \cdot (\text{in} + \text{out})$, capped at $r \cdot (\text{in} + \text{out}) \le \text{in} \cdot \text{out}$ (otherwise "sparse" would be larger than the dense base). The delta matrix is rebuilt by COO scatter and **added** to the base weight (the opposite of HiRA's multiplicative coupling):

$$W' = W + \Delta, \quad \Delta_{ij} = \sum_{k:\ (r_k,c_k)=(i,j)} s \cdot v_k$$

The summation in that formula is the crux of this problem: upstream builds the delta with `torch.sparse_coo_tensor(indices, values, shape)` and calls `to_dense()`, and COO tensors **accumulate duplicate coordinates** — two values at the same position add up rather than the later one overwriting. The mask-to-indices path (upstream `torch.where(mask == 1.0)`) can genuinely produce duplicates, so getting this wrong corrupts the whole Δ. Given coordinate pairs, values, and a scaling factor, reproduce this sparse delta, the additive merge, and the forward pass.

## Task

Implement `solve(weight, indices, values, scaling, x)`:

- `weight`: 2-D float tensor `[out, in]`;
- `indices`: non-empty `list` of `[row, col]` pairs (`list` or `tuple`), integer coordinates with `0 <= row < out`, `0 <= col < in`; duplicates allowed;
- `values`: numbers in a `list` or 1-D tensor, same length as `indices`;
- `scaling`: non-negative finite real (1.0 during training, arbitrary at inference), multiplied into each value;
- `x`: tensor with at least one dimension whose last dim equals `in`.

Return a `dict` with keys `delta_weight` (rebuilt `[out, in]`), `merged_weight` (`weight + delta`), `y` (`x @ merged.T`, shape `[..., out]`).

## Exact rules

1. **Parameter budget**: the coordinate count $K$ must not exceed `in * out` (upstream `update_layer` rejects sparse params outnumbering base elements).
2. **Scatter rebuild**: `delta` starts as a zero matrix; for each coordinate $(r_k, c_k)$ do `delta[r_k, c_k] += scaling * v_k` — **duplicates accumulate**.
3. **Additive merge**: `merged = weight + delta` (contrast HiRA's `W⊙(1+Δ)`; SHiRA writes into zero entries just fine).
4. **Forward**: `y = x @ merged.T` (upstream forward adds the delta into the base weight and runs one `F.linear` on the merged weight).

## Example

```text
W = [[1, 2, 0], [0, 0, 0]] (out=2, in=3)
indices = [[0,0], [0,0], [1,2]], values = [1, 2, 3], scaling = 2
x = [[1, 1, 1]]

(0,0) appears twice: 1*2 + 2*2 = 6 (accumulate; overwriting would give 4)
(1,2) once: 3*2 = 6
delta = [[6, 0, 0], [0, 0, 6]]
merged = [[7, 2, 0], [0, 0, 6]]
y = x @ merged.T = [[9, 6]]
```

Semantic anchor: with an all-zero `W`, coordinate `[[0,1]]`, value 4 — `merged = [[0,4,0],[0,0,0]]` and `y = [[4, 0]]`; under additive semantics the delta lifts entries even where the base is zero (multiplicative HiRA would freeze them forever).

## Constraints

- Do not modify inputs; invalid inputs (weight not 2-D, empty indices / non-pair coordinate / out-of-range or negative coordinate, values length mismatch or non-numeric or non-finite, scaling negative/infinite/bool, x last-dim mismatch or 0-D or non-finite, K over budget) raise `ValueError`.
