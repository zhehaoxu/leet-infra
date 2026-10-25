# Bilinear Upsampling

## Problem

Bilinear sampling treats each channel of the source image as a bilinear function on the grid and evaluates
it at the target pixel coordinates. The core is the **coordinate mapping**, and PyTorch has two conventions:

- `align_corners=False` (default): pixels are treated as area elements, and the target pixel center is mapped
  back to source coordinates
  $$ x_{\text{src}} = (x_{\text{dst}} + 0.5)\cdot\frac{H_{\text{src}}}{H_{\text{dst}}} - 0.5 $$
- `align_corners=True`: the four corners are aligned exactly
  $$ x_{\text{src}} = x_{\text{dst}} \cdot \frac{H_{\text{src}}-1}{H_{\text{dst}}-1} $$

Once the source coordinate $(y, x)$ is known, take the four surrounding integer grid points and blend them by
area weight:

$$
v = (1-w_y)(1-w_x)v_{00} + (1-w_y)w_x v_{01} + w_y(1-w_x)v_{10} + w_y w_x v_{11}
$$

## Rules

- Implement `solve(x, size, align_corners=False)`, where x has shape `(N, C, H, W)`, `size = (oh, ow)`, and
  both upsampling and downsampling are supported.
- `clamp` the source coordinate to $[0, H-1]$ first (the mapping may be negative or out of range), then take
  floor for the lower-left corner and the fractional part as the weight; the upper grid-point indices must be
  clamped as well.
- Do not call `F.interpolate` / `F.grid_sample`; indexing the four neighbors by row/column and broadcasting
  the weights is fully vectorized.
- **Pitfall**: on the same input the two `align_corners` settings differ almost everywhere except at the
  corners; a wrong mapping formula shows up especially clearly when downsampling.

## Example

```
Input: x = [[[1,2],[3,4]]], size=(4,4), align_corners=False
Output: [[1.00, 1.25, 1.75, 2.00],
      [1.50, 1.75, 2.25, 2.50],
      [2.50, 2.75, 3.25, 3.50],
      [3.00, 3.25, 3.75, 4.00]]
```

## Constraints

- float32/float64; $1 \le$ target size, arbitrary up/down scaling factors
- Must match `torch.nn.functional.interpolate(x, size=..., mode="bilinear", align_corners=...)`
