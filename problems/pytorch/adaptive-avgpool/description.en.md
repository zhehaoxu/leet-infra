# Adaptive Average Pooling

## Problem

Adaptive Avg Pooling does not take a window size; instead it takes the **target output size** and derives each output position's sampling interval from the input/output sizes. PyTorch's boundary formulas are:

$$
\text{out}[i] = \text{mean}\Big(x\big[\, \big\lfloor \tfrac{i \cdot H}{o_h} \big\rfloor \;:\; \big\lceil \tfrac{(i+1)\cdot H}{o_h} \big\rceil \,\big] \Big)
$$

The intervals have unequal lengths and may even overlap (e.g. for $H=5, o_h=4$ the slices are $[0,2),[1,3),[2,4),[3,5)$).

## Rules

- Implement `solve(x, output_size)`, where `x` has shape `(N, C, H, W)` and `output_size` is an int or `(oh, ow)`.
- Apply the floor/ceil formulas above independently to height and width, and average over each $(i,j)$ interval.
- Calling `F.adaptive_avg_pool2d` is forbidden; a small loop over output positions plus slicing and `.mean()` is a perfectly valid approach.
- **Pitfall**: naive splitting such as "take a fixed window of $H/o_h$ with rounding" is always wrong for non-divisible sizes — the start must use floor and the end must use ceil.

## Examples

```
Input: x = 4x4 = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]], output_size=2
Output: [[[3.5, 5.5], [11.5, 13.5]]]    # 4->2 divides evenly, so the intervals are exactly equal-width two-row two-column blocks
```

## Constraints

- float32/float64; `output_size` may exceed the input size (most intervals then have length 1)
- The result must match `torch.nn.functional.adaptive_avg_pool2d`
