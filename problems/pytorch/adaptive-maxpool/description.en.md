# Adaptive Max Pooling

## Problem

Ordinary `maxpool2d` uses a fixed `kernel/stride` to determine the output size, so the output changes whenever the input does;
**adaptive** pooling works the other way around — you specify the **desired output size** `output_size`, and the framework derives which part of the input each output position
should cover, then takes the maximum within that segment. It is commonly used to compress feature maps of arbitrary size into a fixed size
(e.g. `AdaptiveMaxPool2d(1)` before a classification head), or, complementarily to `adaptive-avgpool`, to preserve salient activations.

## Rules

- Implement `solve(x, output_size)`, where `x` has shape `(N, C, H, W)` and `output_size` is an integer or `(oh, ow)`.
- The boundaries of the input segment covered by each output position follow PyTorch's floor/ceil formulas:
  output `i` corresponds to the input interval `[floor(i*Hin/outH), ceil((i+1)*Hin/outH))` (likewise for width).
- Take the `max` over all elements in the segment; the output shape is `(N, C, oh, ow)`.
- The boundary slicing is **exactly the same** as in `adaptive-avgpool`; only the aggregation changes from mean to maximum.

## Examples

```
Input: x = [[ 0,  1,  2,  3],
          [ 4,  5,  6,  7],
          [ 8,  9, 10, 11],
          [12, 13, 14, 15]], output_size = 2
   The four 2x2 segments take max respectively: max{0,1,4,5}=5, max{2,3,6,7}=7,
                         max{8,9,12,13}=13, max{10,11,14,15}=15
Output: [[ 5,  7],
      [13, 15]]
```

## Constraints

- float32 input; each dimension of `output_size` must be positive and no larger than the corresponding input dimension (generally satisfied)
- When `output_size == 1` the result is the per-channel maximum of the whole map, equivalent to global max pooling
