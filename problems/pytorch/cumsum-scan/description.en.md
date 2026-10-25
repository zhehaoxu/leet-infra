# Cumulative Sum Scan (Cumsum / Prefix Scan)

## Problem

The cumulative sum (prefix sum) outputs, along some axis, the "total so far": position `i` equals the accumulation of `0..i` along that axis.
It is the most basic of the scan-style operators and is widely used to turn a reduction into "keeping the intermediate result at every step" — for example
computing a prefix sum over probabilities in top-p sampling to find the cutoff point, constructing Transformer relative positions/causal masks, and
writing the recurrent state as a parallel scan in linear attention / SSMs (Mamba). Implement it **without `torch.cumsum`**.

## Rules

- Implement `solve(x, axis)` and return the cumulative sum along `axis`, with the same shape as the input.
- `axis` may be negative.
- Recursive definition: `out[0] = x[0]`, `out[i] = out[i-1] + x[i]` (along `axis`).
- The result must match `torch.cumsum(x, dim=axis)`.

## Examples

```
Input: x = [[1, 2, 3],
          [4, 5, 6]]
  axis=1 (prefix sum within each row) → [[1, 3, 6],
                          [4, 9, 15]]
  axis=0 (prefix sum down each column) → [[1, 2, 3],
                            [5, 7, 9]]
```

## Constraints

- Support inputs of any rank; accumulate along `axis`
- Do not use ready-made scan operators such as `torch.cumsum` / `torch.addcmul`
