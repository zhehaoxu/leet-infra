# Group-wise Fake-Quant

## Problem

INT4/INT8 quantization of LLM weights (GPTQ, AWQ) uses neither per-tensor nor whole-row per-channel granularity, but **group-wise** quantization: split each row into fixed-size `group_size` chunks along the input-feature dimension, each with its own scale. Because the dynamic range of a single output channel also fluctuates sharply along the K direction, per-channel is too coarse; grouping strikes a balance between "accuracy" and "scale storage/overhead" (`group_size=128` is a common default). This problem implements group-wise symmetric fake-quant.

## Rules

Implement `solve(W, group_size, bits=8)`, returning the **dequantized float tensor** (same shape and dtype as `W`):

1. Let $q_{max} = 2^{\text{bits}-1} - 1$ (bits=8 → 127, bits=4 → 7).
2. Along the **last dimension**, split each vector into contiguous groups: group 0 covers elements `[0, group_size)`, group 1 covers `[group_size, 2·group_size)`, … The **last group may be shorter than group_size** (it is not dropped; its statistics are computed on its own).
3. For each group $g$: $\text{scale} = \max_{i\in g}|W_i| \,/\, q_{max}$; for that group $q = \operatorname{clamp}(\operatorname{round}(W_i/\text{scale}),\,-q_{max},\,q_{max})$, and write back $q\cdot\text{scale}$.
4. **Degenerate group** (all zeros): absmax=0 → return 0 for that group unchanged (avoid division by zero).

> For `[out, in]` weights this means "each row is grouped independently"; besides `out` there may be leading dimensions (every position other than the last dimension forms its own row of groups).

## Examples

`bits=8, group_size=4`, $q_{max}=127$:

```
W = [[1.0, -2.0, 0.5, 0.25,  4.0, -1.0]]
group 0 absmax=2 → scale=2/127; group 1 [4,-1] absmax=4 → scale=4/127
output = [[1.00787, -2.0, 0.50394, 0.25197,  4.0, -1.00787]]
```

`bits=4, group_size=4`, $q_{max}=7$:

```
W = [[1.0, 3.0, -7.0, 0.5,  2.0, 2.0, 2.0, 2.0]]
group 0 absmax=7 → scale=1 → [1, 3, -7, round(0.5)=0 (banker's)] → [1,3,-7,0]
group 1 absmax=2 → scale=2/7 → each 2→round(7)=7→7·(2/7)=2
output = [[1.0, 3.0, -7.0, 0.0,  2.0, 2.0, 2.0, 2.0]]
```

## Constraints

- `1 ≤ group_size` (it may exceed the length of the last dimension, in which case the whole row is one group); `bits ∈ {2,…,16}`
- `round` uses banker's rounding (same as `torch.round`); do not modify the input, and the output dtype matches the input
- The short tail of the last group must have its scale estimated **over its own length**; do not zero-pad to a full group (padding with zeros would incorrectly inflate absmax)
