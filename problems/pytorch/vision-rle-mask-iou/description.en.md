# Detection: RLE Codec and Mask IoU

## Problem

Instance-segmentation evaluation (COCO mask AP, Mask R-CNN validation) computes an intersection-over-union between two masks at almost every step. Real datasets never store masks densely: a 1024×1024 image may contain a target of a few hundred pixels, so a dense array wastes memory and compresses poorly. COCO instead scans the mask in **column-major** order into a 0/1 sequence and run-length encodes it (RLE): starting from background, `counts` alternates the length of each run of equal values.

So mask IoU becomes a **format** problem rather than a pixel problem: given one dense mask and another mask's RLE, compute the intersection, the union and the IoU, and re-encode the dense mask into canonical RLE. Two traps:

- **Column-major**: the linear order of `counts` is "walk one full column, then the next"; a row-major encoding of the same mask produces a completely different array;
- **Canonical form**: the first entry is the background run (it may be 0), every other entry must be positive, and a trailing background run must not be dropped.

The task is to get both right, and to allow computing the intersection by **merging the two run sequences** without ever expanding `counts` into a dense mask. CPU only, deterministic, no RNG.

## Rules

The complete signature, parameter meanings and defaults live only in the starter docstring, which is the sole judging contract.

- The dense mask is an \( (H,W) \) bool or uint8 tensor with entries 0/1; `size` is a length-2 integer sequence \( (H,W) \) that must equal the mask shape **exactly**, with \(H,W\ge1\).
- `counts` is a 1-D integer tensor holding the COCO RLE of the other mask:
  - traversal is **column-major**, linear index \( \mathrm{lin}=x\cdot H+y \) (\(x\) column, \(y\) row);
  - run values alternate starting from 0 (background); `counts[0]` is the background length before the first foreground pixel (possibly 0), odd indices are foreground runs, even indices background runs;
  - non-empty, all non-negative, **no zero except possibly the first entry** (canonical form, no interior zero-length runs), and the entries sum to \(H\cdot W\).
- Return a 4-tuple:
  - `iou`: 0-dim float64, \( \mathrm{inter}/\mathrm{union} \), 0 when `union` is 0;
  - `intersection`: 0-dim int64, pixels foreground in both masks;
  - `union`: 0-dim int64, pixels foreground in at least one mask, \( =|A|+|B|-|A\cap B| \);
  - `counts_a`: 1-D int64, the **canonical** column-major RLE of the dense mask.
- Canonical encoding details: an all-background mask encodes as `[H*W]`; when the first column-major pixel is foreground the first entry is 0 (e.g. `[0, H*W]` is all foreground); a trailing background run is kept; adjacent equal-value runs are merged, so no interior zeros appear.
- Invalid input raises `ValueError`: mask is not a tensor / not 2-D / dtype is not bool or uint8 / contains values other than 0 and 1; `counts` is not a 1-D integer tensor / empty / negative / contains a zero after the first entry / does not sum to \(H\cdot W\); `size` is not a length-2 integer sequence / non-positive / inconsistent with the mask shape.
- No input tensor is modified.

## Example

With \(H=2,W=3\), the column-major linear index maps to coordinates as

```
lin   :  0    1    2    3    4    5
(x,y) : (0,0)(0,1)(1,0)(1,1)(2,0)(2,1)
```

Mask A (dense) and mask B (given as RLE):

```
A row-major = [[1, 0, 0],
               [0, 0, 1]]
B row-major = [[1, 1, 0],
               [0, 1, 1]]
```

Column-major expansion:

```
A :  1  0  0  0  0  1   ->  counts_a = [0, 1, 4, 1]
B :  1  0  1  1  0  1   ->  counts   = [0, 1, 1, 2, 1, 1]
```

Reading `counts_a`: the leading 0 means "the very first pixel is foreground", then 1 foreground pixel, 4 background pixels, 1 foreground pixel, total 6. Reading `counts`: 0 background, 1 foreground, 1 background, 2 foreground, 1 background, 1 foreground, total 6.

The intersection sits at `lin=0` and `lin=5`, hence

```
intersection = 2
|A| = 2, |B| = 4  ->  union = 2 + 4 - 2 = 4
iou = 2 / 4 = 0.5
```

A's row-major and column-major flattenings happen to coincide here (both `[1,0,0,0,0,1]`), so this example cannot separate the two orders. A discriminating one: the mask `[[1,1,0],[0,0,0]]` flattens column-major to `[1,0,1,0,0,0]` and encodes as `[0,1,1,1,3]`, while row-major flattening gives `[1,1,0,0,0,0]` and encodes as `[0,2,4]`. Interpreting `[0,2,4]` as a column-major RLE leaves only 1 intersection pixel with A, so the IoU is \(1/3\) instead of 1.

## Constraints

- The return order is fixed as `(iou, intersection, union, counts_a)`; `iou` is 0-dim float64, `intersection`/`union` are 0-dim int64, `counts_a` is 1-D int64.
- The judge uses absolute tolerance \( 10^{-12} \) for `iou` (values are exact integer ratios in float64) and `torch.equal` for `intersection`, `union` and `counts_a`.
- When both masks are empty, `union=0` and `iou` is defined as 0 rather than NaN.
- CPU only, deterministic, no RNG; whether `counts` is int32 or int64 and whether the mask is bool or uint8 must not change the result.
