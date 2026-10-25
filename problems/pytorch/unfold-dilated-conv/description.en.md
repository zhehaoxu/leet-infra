# Unfold Dilated Patch Extraction

## Problem

The im2col view of convolution: extract the receptive-field "patch" corresponding to every output position
on the input feature map. With a dilation rate `dilation` $d$, adjacent samples inside a patch are $d$
pixels apart in the original image (skipping $d-1$ pixels in between).

$$ \text{block}[r,w,i,j] = x[..., \; r + i\cdot d,\; w + j\cdot d],\quad 0 \le i < k_H,\ 0 \le j < k_W $$

Number of output positions (stride=1, no padding):

$$ o_H = H - (k_H - 1)d, \qquad o_W = W - (k_W - 1)d $$

## Rules

- Implement `solve(x, kernel_size, dilation=1)`, where `x` has shape `(B, C, H, W)` and
  `kernel_size=(kH, kW)`.
- Return a tensor of shape `(B, C, oH, oW, kH, kW)`: the last two dims are the $k_H \times k_W$ samples inside
  each patch, following the dilated sampling formula above.
- `dilation=1` degenerates into ordinary contiguous patch extraction.
- `Tensor.unfold` may be used: first unfold a contiguous window with span $(k-1)d+1$, then sample inside the
  window with step $d$.

## Example

```
Input: x = 1×1×3×4, values [[0,1,2,3],[4,5,6,7],[8,9,10,11]], kernel=(2,2), dilation=2
   oH=3-(2-1)*2=1, oW=4-2=2 → shape (1,1,1,2,2,2)
   patch(0,0) samples rows{0,2}×cols{0,2} = [[0,2],[8,10]]
   patch(0,1) samples rows{0,2}×cols{1,3} = [[1,3],[9,11]]
```

## Constraints

- float32/float64 input; $(k_H-1)d < H$, $(k_W-1)d < W$
