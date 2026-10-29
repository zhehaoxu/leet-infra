# repeat_interleave: Element-wise Repetition via index_select

## Problem Description

`repeat_interleave` and `repeat`/`tile` do different things yet are the most frequently confused pair: repeating `[1,2,3]` twice gives `[1,1,2,2,3,3]` in the first case (each element expanded into its own block) and `[1,2,3,1,2,3]` in the second (the whole sequence tiled). Expanding KV heads for GQA, growing a token-level mask into token×head, duplicating RoPE cos/sin pairwise along head_dim, and rebuilding a variable-length sequence from the `cu_seqlens` deltas all use the former.

$$\text{out}\Big[\sum_{j<i} r_j + k\Big] = x[i], \qquad 0 \le k < r_i$$

i.e. output position $\sum_{j<i} r_j + k$ is taken from element $i$ of the input along that axis, so the cheapest equivalent implementation is **build the index vector first, then `index_select`**.

## Rules

- The full signature and parameter semantics live only in the starter docstring; that is the single source of truth for judging.
- `repeats` is a non-negative int or a **1-D integer tensor** (a per-position count whose length must equal the number of elements along `dim`); negatives, float tensors, 2-D tensors and length mismatches all raise ValueError.
- `dim=None`: flatten `x` to 1-D first, repeat elementwise and return a 1-D tensor of length $=\sum_i r_i$; with `dim=int` only that axis is expanded and every other axis is preserved, negatives folded against `x.dim()`, out of range raising ValueError.
- A 0 inside `repeats`: that position disappears entirely (the output length may be 0, in which case an **empty tensor with the same number of axes and dtype** is returned).
- Calling `torch.repeat_interleave` directly is forbidden; the output must be freshly allocated memory (the judge calls `mul_` on it and asserts the input is untouched).

## Example

$X=[[1,2],[3,4]]$ (numbers taken from the EXAMPLE_* constants filled in from test_cases.py):

```
X                    = [[1, 2],
                        [3, 4]]

solve(X, 2)          = [1, 1, 2, 2, 3, 3, 4, 4]     # dim=None flattens first
solve(X, 2, dim=0)   = [[1, 2], [1, 2], [3, 4], [3, 4]]
solve(X, [1, 3], 0)  = [[1, 2], [3, 4], [3, 4], [3, 4]]
solve([5, 6, 7], [2, 0, 1]) = [5, 5, 7]
```

Row 3 has per-position counts: row 0 of the input repeats once, row 1 repeats three times. In the last row `6` has count 0 and disappears entirely — the output length is $\sum r_i = 3$.

## Constraints

- Input is float32; the output dtype and number of axes match the input, and results must be **exactly equal** to `torch.repeat_interleave` (`torch.equal`, no floating tolerance)
- `repeats` covers int, a random long tensor, int32 and an all-zero tensor; `dim` covers None, positive, negative and middle axes
- Each illegal argument (negative count, float repeats, 2-D repeats, length mismatch, out-of-range dim) must raise ValueError
