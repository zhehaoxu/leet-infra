# Intersection over Union (IoU)

## Problem

IoU (also called the Jaccard index) measures the overlap between two rectangular boxes: the intersection area divided by the union area, in the range $[0,1]$.
It is a cornerstone of object detection/segmentation—matching predicted boxes with ground-truth boxes, suppressing overlapping boxes in NMS, and deciding TP/FP for
mAP all use IoU as the criterion. Please implement pairwise IoU.

## Rules

- Implement `solve(a, b)`, with boxes in $[x_1, y_1, x_2, y_2]$ form (top-left + bottom-right), shape `(..., 4)`, broadcastable.
- Intersection width and height: $w=\max(0,\min(x_{2a},x_{2b})-\max(x_{1a},x_{1b}))$ (same for $y$), with negative values clamped to 0.
- $\text{IoU} = \dfrac{\text{intersection}}{\text{area}_a + \text{area}_b - \text{intersection}}$; return 0 when the union is 0.
- The output shape is the broadcast shape with the last dimension removed.

## Examples

```
Input: a = [0, 0, 2, 2] (area 4)
     b = [1, 1, 3, 3] (area 4)
  intersection = [1,1,2,2] → 1×1 = 1; union = 4 + 4 - 1 = 7
Output: 1/7 ≈ 0.142857
```

## Constraints

- float32 input; coordinates satisfy $x_2\ge x_1, y_2\ge y_1$ (a degenerate zero-area box returns 0 when the union is 0)
- Perfect overlap → 1; no overlap at all → 0
