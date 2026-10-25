# Bilinear Grid Sampling Explained

## Normalized coordinates → pixel coordinates

`grid` expresses positions in `[-1,1]`, and there are two alignment conventions between that and pixel indices:

- `align_corners=True`: $x_{pix}=\tfrac{g+1}{2}(W-1)$, so the corners $-1/1$ fall exactly on the first/last pixel centers.
- `align_corners=False` (this problem): $x_{pix}=0.5(g+1)W-0.5$, which treats the whole image as the continuous region $[0,W]$
  with the corners on the **outer edges** of the pixels. At $g=0$, $x_{pix}=\tfrac{W-1}{2}$, exactly the geometric center of the image.

The two conventions differ noticeably when $W$ is even; PyTorch's modern default is `False` (more consistent for image augmentation / pyramids).

## Bilinear interpolation formula

Let $x_0=\lfloor x_{pix}\rfloor,\ f_x=x_{pix}-x_0$ (same for $y$); the four integer neighbors
$(x_0,y_0),(x_0{+}1,y_0),(x_0,y_0{+}1),(x_0{+}1,y_0{+}1)$ are weighted as:

$$ v = (1{-}f_y)\big[(1{-}f_x)a + f_x b\big] + f_y\big[(1{-}f_x)c + f_x d\big] $$

Two linear interpolations along $x$ followed by one along $y$ is equivalent to a separable interpolation over the 2×2 neighborhood.

## Out-of-bounds and padding_mode

`zeros`: any neighbor outside the image contributes 0 (in practice, pad `x` with a ring of zeros and shift all indices by +1 so
lookups naturally land in the zero-padded region). There are also `border` (clamp to the edge) and `reflection` (mirror). With values in `[-1,1]`
you can overstep by at most one and a half pixels, so a single ring of padding suffices and there is no need to worry about farther indices.

## Why it is differentiable everywhere / which operators it underpins

The output is a fixed-weight linear combination with respect to `x`, and differentiable with respect to `grid` through $f_x,f_y$ (except at
integer boundaries), so gradients flow back to the coordinates themselves — exactly why **STN** (which learns an affine matrix to generate the grid) and
**deformable convolution DCN** (which adds learned offsets to a regular grid and then grid_samples) can bring "geometry" into end-to-end
training. Separable convolution and bilinear downsampling are special cases of it under particular grids.
