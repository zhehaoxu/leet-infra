# Depthwise Separable Convolution

## Problem

Depthwise separable convolution splits a standard convolution into two steps: **depthwise** (each channel uses its own $k\times k$ kernel and is convolved only spatially)
+ **pointwise** (a $1\times1$ convolution that linearly combines across channels). It reduces the compute from $C_{in}C_{out}k^2$ to
$C_{in}k^2 + C_{in}C_{out}$, and is the core of MobileNet / Xception / EfficientNet.

## Rules

- Implement `solve(x, dw_weight, pw_weight, dw_bias=None, pw_bias=None)`, where `x` has shape `(N, C, H, W)`.
- **Step 1, depthwise**: `dw_weight` has shape `(C, 1, kh, kh)` (`kh` is odd), `stride=1`,
  `padding = kh//2` (keeping the spatial size unchanged), and each channel uses its own kernel; producing `(N, C, H, W)`, then add `dw_bias`.
- **Step 2, pointwise**: `pw_weight` has shape `(Cout, C, 1, 1)`; a $1\times1$ convolution linearly combines the `C` channels
  into `Cout` (no padding), then add `pw_bias`; the output is `(N, Cout, H, W)`.

## Examples

```
Input: x=all ones (1,2,3,3); dw_weight=all ones (2,1,3,3) (padding=1)
     pw_weight=[[[0],[1]],[[2],[3]],[[4],[5]]] shape (3,2,1,1)
   after depthwise the center=9 (sum of the all-ones 3x3), corners=4, edges=6
   pointwise: out0=0*dw0+1*dw1, out1=2*dw0+3*dw1, out2=4*dw0+5*dw1
Output: (1,3,3,3)
  ch0=[[4,6,4],[6,9,6],[4,6,4]]
  ch1=[[20,30,20],[30,45,30],[20,30,20]]
  ch2=[[36,54,36],[54,81,54],[36,54,36]]
```

## Constraints

- float32 input; `kh` is odd; `dw_bias`/`pw_bias` may be `None`
- Equivalent to `F.conv2d(F.conv2d(x, dw_weight, dw_bias, padding=kh//2, groups=C), pw_weight, pw_bias)`
