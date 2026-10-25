# Leaky ReLU

## Problem

ReLU's gradient is identically 0 in the negative region (the "dead neuron" problem). Leaky ReLU gives the negative region a small slope so gradients survive:

$$
\text{LeakyReLU}(x) = \max(0, x) + \text{negative\_slope} \cdot \min(0, x) =
\begin{cases} x, & x \ge 0 \\ \text{negative\_slope} \cdot x, & x < 0 \end{cases}
$$

## Rules

- Implement `solve(x: torch.Tensor, negative_slope: float = 0.01) -> torch.Tensor`.
- The output shape and dtype match the input; `negative_slope=1` degenerates to the identity and `=0` degenerates to ReLU.
- A vectorized implementation is required (e.g. `torch.where`); Python loops are not allowed.

## Examples

```
Input: x = [-2.0, -1.0, 0.0, 1.0, 2.0], negative_slope = 0.01
Output: [-0.02, -0.01, 0.0, 1.0, 2.0]
```

## Constraints

- The input is a float32 or float64 tensor of arbitrary shape
- `negative_slope` is a non-negative floating-point scalar
