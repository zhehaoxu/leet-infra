# 3D Convolution (conv3d)

## Problem

3D convolution slides the convolution kernel simultaneously over **three** spatial dimensions $(D, H, W)$; the input looks like voxel data,
video frame sequences, or medical images (CT/MRI). Each output element is the sum of element-wise products between the kernel and a 3D window
over all input channels. As with conv1d/conv2d, "convolution" here is **cross-correlation** (the kernel is not flipped).

## Rules

- Implement `solve(x, weight, bias, stride=1, padding=0)`, where `x` has shape `(N, Cin, D, H, W)`.
- `weight` has shape `(Cout, Cin, kD, kH, kW)` and `bias` has shape `(Cout,)` (it may also be `None`).
- For simplicity, `stride` and `padding` are scalars in this problem, shared by all three spatial dimensions. The output length along each dimension:
  $$ L_{out} = \left\lfloor \frac{L + 2\cdot\text{padding} - k_L}{\text{stride}} \right\rfloor + 1,\quad L \in \{D,H,W\} $$
- Each output element:
  $$ \text{out}[n,o,a,b,c] = \text{bias}[o] + \sum_{ci}\sum_{di}\sum_{hi}\sum_{wi} x[n, ci, a\cdot s+di, b\cdot s+hi, c\cdot s+wi]\cdot w[o, ci, di, hi, wi] $$
- First pad `padding` zeros on both sides of each of the three spatial dimensions, then slide with `stride`. The result must match
  `F.conv3d(x, weight, bias, stride=stride, padding=padding)`.

## Examples

```
Input: x = arange(1..8) reshape (1,1,2,2,2)   (voxel values 1,2,...,8)
     weight = all ones, shape (1,1,2,2,2)
     bias = [0], stride=1, padding=0
   output size (2-2)//1+1 = 1, sum over the whole 2x2x2 window = 1+2+...+8 = 36
Output: [[[[[36]]]]]                            shape (1,1,1,1,1)

If padding=1: output size (2+2-2)//1+1 = 3 → shape (1,1,3,3,3)
```

## Constraints

- float32 input; `stride >= 1`, `padding >= 0` (scalars in this problem)
- The input-channel dimension of `weight` must equal the number of channels `Cin` of `x` (no grouping)
- Do not use `F.conv3d` / `torch.nn.Conv3d`; implement the 3D sliding window by hand
