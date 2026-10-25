# Grouped Convolution

## Problem

Grouped convolution splits the input channels into `groups` segments; each segment is convolved **only over its own subset of input channels** and produces
`Cout//groups` output channels, with no connections between groups. It reduces parameters and compute proportionally,
and is a key building block of networks such as ResNeXt and MobileNet (depthwise is `groups=C`).

## Rules

- Implement `solve(x, weight, bias, groups)`, where `x` has shape `(N, Cin, H, W)`.
- `weight` has shape `(Cout, Cin//groups, kh, kw)` (one input-channel dimension smaller than a standard convolution by a factor of `groups`).
- `stride=1`, `padding=0`; the output shape is `(N, Cout, H-kh+1, W-kw+1)`; output channels are split contiguously by group.
- Both `Cin` and `Cout` are divisible by `groups`; group `g` uses input channels `[g*(Cin//groups):(g+1)*(Cin//groups)]`
  to produce output channels `[g*(Cout//groups):(g+1)*(Cout//groups)]`.

## Examples

```
Input: x is (1,4,3,3), channels 0/1/2/3 are filled entirely with 1/2/3/4 respectively
     weight=all ones, shape (4,2,2,2); bias=0; groups=2
   group0 (channels 0,1): sum over each 2x2 window = 4*(1+2) = 12 → output channels 0,1
   group1 (channels 2,3): sum over each 2x2 window = 4*(3+4) = 28 → output channels 2,3
Output: out[0]=out[1]=[[12,12],[12,12]], out[2]=out[3]=[[28,28],[28,28]]
```

## Constraints

- float32 input; `groups >= 1`, `groups` divides `Cin` and `Cout`
- `groups == 1` must match a standard convolution; `groups == Cin` is depthwise convolution
