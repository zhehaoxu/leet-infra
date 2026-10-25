# Bilinear Grid Sample

## Problem

`grid_sample` is a general operator for "differentiably reading values from an image at arbitrary coordinates": given a feature map `x` and a
normalized coordinate grid `grid`, it outputs the value obtained by **bilinear interpolation** of `x` at each grid point. It is the
core primitive behind STN (Spatial Transformer Networks), deformable convolution (DCN), image warping, cross-resolution feature
alignment, and more — because it is differentiable with respect to both `x` and `grid`, it lets geometric transforms participate in end-to-end training.

## Rules

- Implement `solve(x, grid)`: `x` has shape `(N, C, H, W)`, `grid` has shape `(N, Ho, Wo, 2)`.
- `grid[..., 0]` is the normalized coordinate along the **width (x)** direction and `grid[..., 1]` along the **height (y)** direction, in the range `[-1, 1]`;
  `(-1,-1)` maps to the top-left pixel center and `(1,1)` to the bottom-right pixel center.
- Use the `align_corners=False` coordinate transform (mapping to pixel coordinates):
  $$ x_{pix} = 0.5\,(g_x+1)\,W - 0.5,\qquad y_{pix} = 0.5\,(g_y+1)\,H - 0.5 $$
- For each output position, take the four surrounding integer pixels and apply bilinear weighting; sampling points that fall outside the image are padded with **0**
  (`padding_mode='zeros'`).
- The output has shape `(N, C, Ho, Wo)`; all channels are sampled with the same set of coordinates.

## Examples

```
Input: x = [[0, 1, 2],
            [3, 4, 5],
            [6, 7, 8]]   (1,1,3,3)
     grid four points (in order Ho=2,Wo=2): (0,0), (-1,-1), (1,1), (0.5,0.5)
Output: [[4.0, 0.0],        # (0,0)→the exact center pixel 4; (-1,-1)→outside the top-left corner, zero-padded
        [2.0, 7.0]]        # (1,1)→outside the bottom-right corner, 8*0.5*0.5=2; (0.5,0.5)→bilinear gives 7
```

## Constraints

- float32 input; `grid` values in `[-1, 1]` (already clamped by the problem)
- Must match `F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)`
  within `atol=1e-4`
