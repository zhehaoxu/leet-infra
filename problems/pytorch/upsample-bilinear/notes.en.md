# Bilinear Upsampling / Downsampling In Depth

## Definition

Resample $(N,C,H,W)$ to the target size; each new pixel is a weighted average of the **4 neighbors** around
its source coordinate:

$$ \text{out}(y,x)=(1-w_y)(1-w_x)v_{00}+(1-w_y)w_x v_{01}+w_y(1-w_x)v_{10}+w_y w_x v_{11} $$

The weights are determined by the fractional part of the source coordinate; interpolate linearly along rows
first and then along columns (or the other way round — equivalent).

## The core trap: `align_corners` (two coordinate mapping formulas)

Where the $i$-th output pixel maps back to in the input depends on this switch:

- **`align_corners=True`**: aligns **corner to corner**, scale $=\dfrac{H_{in}-1}{H_{out}-1}$,
  $\text{src}=i\cdot\text{scale}$. The two endpoints land exactly on the corner input pixels.
- **`align_corners=False` (torch default)**: aligns by pixel **center**, scale $=\dfrac{H_{in}}{H_{out}}$,
  $\text{src}=(i+0.5)\cdot\text{scale}-0.5$. Keeps the "pixel size" consistent, but the corners do not line up.

The two give different numbers everywhere; the judge must use the same convention. Forgetting the `-0.5` or
using the wrong scale is the most common bug.

## Boundary handling

- `clamp` the source coordinate to $[0,H_{in}-1]$; $y_1=(y_0+1).\text{clamp}(\max=H_{in}-1)$ — edge rows/columns
  are repeated to avoid out-of-range access.
- size may be an int (square) or `(oh,ow)`.

## Relation to other interpolations

nearest (take the closest) and bicubic (16 neighbors) are resamplings of the same kind, differing only in the
kernel. Transposed convolution (`ConvTranspose`) and `F.interpolate` cover all these modes; hand-writing
bilinear here is the best way to understand the essence of "coordinate mapping + weights".
