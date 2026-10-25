# Nearest-Neighbor Upsampling

## Problem

Nearest-neighbor upsampling enlarges a feature map by an integer factor: every pixel of the output tensor
simply **copies** the input pixel it maps to, with no interpolation at all. It is the cheapest resize, and is
commonly used in segmentation/detection to restore a low-resolution mask back to the original image size, as
well as in pixel art and quantized image enlargement, where no new gray levels should be introduced.

## Rules

- Implement `solve(x, scale)`, where `x` has shape `(N, C, H, W)` and `scale` is an integer enlargement factor
  (or `(sh, sw)`).
- Output shape is `(N, C, H*scale, W*scale)`.
- The mapping uses the same **floor** rule as `F.interpolate(mode='nearest', scale_factor=scale)`:
  output row `i` comes from input row `i // scale` (columns likewise).
- Every input pixel is copied exactly into a `scale × scale` block.

## Example

```
Input: x = [[1, 2],
          [3, 4]], scale = 2
Output: [[1, 1, 2, 2],
      [1, 1, 2, 2],
      [3, 3, 4, 4],
      [3, 3, 4, 4]]
```

## Constraints

- float32 input; `scale >= 1` integer
- The output is a "blocky enlargement" of the input; the set of values is exactly the same as the input's (no
  new values are produced)
