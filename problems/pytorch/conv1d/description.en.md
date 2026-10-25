# 1D Convolution (conv1d)

## Problem

1D convolution slides a 2D convolution along a single spatial (sequence) dimension: the kernel moves step by step along the length direction and, for each output position, multiplies and sums all input channels against the corresponding kernel elements. It is widely used for time-series signals, 1D-CNNs, and audio/text feature extraction. Here "convolution" follows the deep learning convention, i.e. **cross-correlation** (the kernel is not flipped).

## Rules

- Implement `solve(x, weight, bias, stride=1, padding=0)`, where `x` has shape `(N, Cin, L)`.
- `weight` has shape `(Cout, Cin, kL)` and `bias` has shape `(Cout,)` (it may also be `None`).
- The output length is $L_{out} = \left\lfloor \dfrac{L + 2\cdot\text{padding} - k_L}{\text{stride}} \right\rfloor + 1$,
  and the output shape is `(N, Cout, L_out)`.
- Each output element:
  $$ \text{out}[n,o,t] = \text{bias}[o] + \sum_{ci=0}^{C_{in}-1} \sum_{j=0}^{k_L-1} x\big[n, ci, t\cdot\text{stride}+j\big]\cdot w[o, ci, j] $$
- First zero-pad both sides of the length dimension according to `padding`, then slide with `stride`. The result must match
  `F.conv1d(x, weight, bias, stride=stride, padding=padding)`.

## Examples

```
Input: x = [[[1, 2, 3, 4, 5]]]           shape (1,1,5)
     weight = [[[1, 2, 3]]]             shape (1,1,3)
     bias = [0], stride=1, padding=0
   out[0] = 1*1 + 2*2 + 3*3 = 14
   out[1] = 2*1 + 3*2 + 4*3 = 20
   out[2] = 3*1 + 4*2 + 5*3 = 26
Output: [[[14, 20, 26]]]                  shape (1,1,3)
```

## Constraints

- float32 input; `stride >= 1`, `padding >= 0`
- The input-channel dimension of `weight` must equal the number of channels `Cin` of `x` (no grouping in this problem)
- Do not use `F.conv1d` / `torch.nn.Conv1d`; implement the sliding window by hand
