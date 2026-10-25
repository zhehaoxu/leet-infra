# im2col: Accelerating Convolution

## Problem

im2col is the classic engineering trick that "flattens" convolution into a matrix multiplication: unroll the receptive field of every output position into a single row, producing the column matrix $A \in \mathbb{R}^{(N \cdot o_h o_w) \times (C_{in} k_h k_w)}$; reshape the weights into $W \in \mathbb{R}^{(C_{in} k_h k_w) \times C_{out}}$, then

$$
\text{out}_{\text{flat}} = A \cdot W
$$

A single GEMM replaces what would otherwise be tens of thousands of loop iterations. It is the skeleton of early cuDNN convolutions and the key to understanding "why optimizing GEMM kernels is equivalent to optimizing convolution".

## Rules

- Implement `solve(x, weight, bias=None, stride=1, padding=0)`, with shapes as in "2D convolution forward": x $(N, C_{in}, H, W)$, weight $(C_{out}, C_{in}, k_h, k_w)$.
- Pipeline: 0-padding → collect each receptive field with `unfold` (or loops) into $(N, o_h o_w, C_{in} k_h k_w)$ → one matrix multiplication with the reshaped weights → reshape/permute back to $(N, C_{out}, o_h, o_w)$.
- The matrix multiplication may only use `@`/`matmul` in the final step (this is precisely the semantic of the problem: convolution → GEMM); the collection of receptive fields must be hand-written.
- **Trap**: the element order of an unrolled row must be "channel → row → column" (matching the flattening order of the weight's $(C_{in}, k_h, k_w)$); the two matrices must be flattened in the same order to line up.
- Do not call `F.conv2d`.

## Examples

```
Input: x = single channel 3x3 = [[1,2,3],[4,5,6],[7,8,9]], weight = one 2x2 all-ones kernel
col matrix (4 flattened 2x2 receptive fields): [[1,2,4,5],[2,3,5,6],[4,5,7,8],[5,6,8,9]]
Output: col @ ones(4x1) = [[12,16,24,28]] → [[[12,16],[24,28]]]
```

## Constraints

- float32/float64; test sizes are small (the im2col memory blow-up of $k_h k_w C_{in}$× is acceptable)
- The result must match `torch.nn.functional.conv2d`
