# 2D Convolution Forward (Naive Implementation)

## Problem

Convolution is the core of CNNs: every output position and every output channel is an element-wise multiply-add of the input receptive field and the convolution kernel (strictly speaking, **cross-correlation** — PyTorch's `conv2d` performs no flipping):

$$
\text{out}[n,o,i,j] = \sum_{c=0}^{C_{in}-1}\ \sum_{u=0}^{k_h-1}\ \sum_{v=0}^{k_w-1} x_{\text{pad}}[n,c,\,i\cdot s_h+u,\,j\cdot s_w+v]\cdot w[o,c,u,v] + \text{bias}[o]
$$

The output size is $\left\lfloor \frac{H + 2p - k_h}{s_h} \right\rfloor + 1$.

## Rules

- Implement `solve(x, weight, bias=None, stride=1, padding=0)`: `x` has shape $(N, C_{in}, H, W)$ and `weight` has shape $(C_{out}, C_{in}, k_h, k_w)$.
- First pad with **0** according to `padding`, then accumulate per the formula. This problem allows triple loops (output position × output channel; tensor slicing and summing over channels is fine), and performance is not required.
- Do not call ready-made convolution routines such as `F.conv2d` / `nn.Conv2d` / `conv_transpose`.
- **Pitfall**: dimension 0 of `weight` is the output channel and dimension 1 is the input channel — transposing them is the number one mistake when hand-writing this; padding fills with 0 (the opposite of maxpool's $-\infty$).

## Examples

```
Input: x = single-channel 3x3 = [[1,2,3],[4,5,6],[7,8,9]], weight = one 2x2 all-ones kernel
Output: [[[12, 16], [24, 28]]]     # sum over each 2x2 window: 1+2+4+5=12, ...
```

## Constraints

- float32/float64; test inputs are small (naive loops are acceptable)
- The result must match `torch.nn.functional.conv2d`
