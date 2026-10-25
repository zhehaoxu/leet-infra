# Transposed Convolution / Deconvolution (ConvTranspose2d)

## Problem

Transposed convolution, often called "deconvolution", is used to **upsample** feature maps to a larger size (semantic segmentation decoder heads, generators, differentiable resizing).
It is not the inverse of an ordinary convolution, but rather the step of an ordinary convolution that **computes the gradient with respect to the input**: each input element multiplies its value by the kernel and
**scatter-adds** it to several output positions.

$$ o_h = (H-1)\cdot \text{stride} - 2\cdot\text{padding} + k_h $$

## Rules

- Implement `solve(x, weight, bias, stride, padding)`, where `x` has shape `(N, Cin, H, W)`.
- **Note that the weight convention is the opposite of conv2d**: shape `(Cin, Cout, kh, kw)`.
- For each input element `x[n,ci,iy,ix]`, accumulate `weight[ci,:,ky,kx]` into the output position
  `[n,:, iy*stride - padding + ky, ix*stride - padding + kx]`, **discarding out-of-bounds contributions**.
- Finally add the bias per `Cout`. The output shape is `(N, Cout, o_h, o_w)` (see the formula above).

## Examples

```
Input: x=[[[[1,2],[3,4]]]] (1,1,2,2), weight=all ones (1,1,2,2), bias=None, stride=1, padding=0
      each input value adds once to each output in its 2x2 neighborhood; overlaps accumulate
Output: [[[[1, 3, 2],
        [4, 10, 6],
        [3, 7, 4]]]]   (1,1,3,3)
      (center 10 = 1+2+3+4, exactly because all four inputs cover that output position)
```

## Constraints

- float32 input; `stride >= 1`, `padding >= 0`; `bias` may be `None` or have shape `(Cout,)`
- Must support all combinations of `stride ∈ {1,2}` and `padding ∈ {0,1}` and match `F.conv_transpose2d`
