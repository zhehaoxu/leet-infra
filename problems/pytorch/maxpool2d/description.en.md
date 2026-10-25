# 2D Max Pooling

## Problem

Max Pooling takes the maximum inside each $k_h \times k_w$ window and is the workhorse of CNN downsampling (it keeps the strongest response and provides translation invariance):

$$
\text{out}[n,c,i,j] = \max_{0 \le u < k_h,\ 0 \le v < k_w} x\!\left[n,c,\, i\cdot s_h + u,\, j\cdot s_w + v\right]
$$

The output size is $\left\lfloor \frac{H + 2p_h - k_h}{s_h} \right\rfloor + 1$ (the same holds for the width).

## Rules

- Implement `solve(x, kernel_size, stride=None, padding=0)`, where `x` has shape `(N, C, H, W)`; the three parameters accept either an int or an `(h, w)` tuple, and `stride=None` means it equals `kernel_size`.
- **Padding trap**: the pad value for max pooling must be $-\infty$. Padding with 0 raises the maximum of boundary outputs to 0 incorrectly when the input is entirely negative.
- Calling `F.max_pool2d` / `nn.MaxPool2d` directly is forbidden; you may use `unfold` to extract windows followed by `amax`, or write a loop.

## Examples

```
Input: x = 1x1 channel 4x4 = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]], kernel=2
Output: [[[6, 8], [14, 16]]]
```

## Constraints

- float32/float64; the output size follows the floor formula (ceil_mode is not supported)
- The result must match `torch.nn.functional.max_pool2d`
