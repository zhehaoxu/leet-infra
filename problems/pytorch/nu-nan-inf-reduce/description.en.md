# NaN Reduction: Skipping, Counting and All-NaN Segments

## Problem Description

`torch.sum` splashes NaN over the whole segment as soon as it meets one, while `torch.nanmean` only deigns to emit a warning plus a NaN when the segment is entirely NaN. The real requirement is usually a third thing: positions outside the mask hold NaN (keys masked out in attention, missing samples in a sparse batch, ranks that never flushed during a multi-device gather) and must be **skipped while the rest keeps contributing**, with "this segment has no valid element at all" defined separately. This problem asks for four skip-style reductions written by hand, with inf still a legitimate value that participates — plenty of people use the segment length as the `nanmean` denominator, or assume `nanmax` also excludes inf.

## Rules

- The full signature and parameter semantics live only in the starter docstring; that is the single source of truth for judging.
- Write `seg` for one segment under reduction and `valid = the elements of seg that are not NaN`:
  - `sum` = \(\sum\) valid, and **an all-NaN segment gives 0.0** (NaN is dropped as an addend, it does not pollute);
  - `mean` = \(\sum\) valid / **len(valid)**, the denominator is the valid count rather than the segment length, an all-NaN segment gives NaN;
  - `max` / `min` = max/min over valid, an all-NaN segment gives NaN.
- ±inf is **not** NaN and participates normally: a segment holding +inf has max = +inf, a segment holding both ±inf has min = -inf and sum = NaN (inf - inf).
- A reduced axis of length 0 (empty tensor) counts as count == 0: `sum` gives 0.0, everything else gives NaN.
- `dim=None` reduces everything and returns a 0-D tensor; `dim` accepts negative axes and raises ValueError when out of range; `mode` accepts only the four names, `keepdim` only bools; non-float input raises ValueError.

## Example

```
x = [[1, nan, 3], [nan, nan, 2], [nan, nan, nan]]

dim=1  mean -> [2, 2, nan]     sum -> [4, 2, 0]    max -> [3, 2, nan]
dim=0  mean -> [1, nan, 2.5]   sum -> [1, 0, 5]    max -> [1, nan, 3]
dim=None  sum -> 6   mean -> 2   max -> 3   min -> 1
```

Row 0 has 1 and 3 as its valid entries, so `mean` is \((1+3)/2=2\) and not \(4/3\); row 2 has no valid entry at all, so `sum` gives 0.0 while `mean`/`max` give NaN. Going down the columns: column 0 keeps only the single 1, column 1 is all NaN, column 2 holds 3 and 2, which yields `[1, nan, 2.5]` for `dim=0` `mean`. A row containing inf, `[inf, nan, -inf]`: `max` is +inf, `min` is -inf, and `sum` is NaN.

## Constraints

- Any number of dimensions, float32/float64, output dtype matches the input
- the oracle is a pure-Python segmented reduction (slice with movedim, filter out NaN, then the builtin `sum/max/min`), unlike any tensor reduction implementation
- 1e-12 relative tolerance on well-conditioned data, bitwise equality on the example (including NaN positions that must both be NaN)
