# ReLU

## Problem

Implement the elementwise ReLU (Rectified Linear Unit) activation, the most basic operator in deep learning:

$$
\text{ReLU}(x) = \max(0, x)
$$

It is a building block of every modern activation such as GELU and Swish, and the starting point for understanding backpropagation.

## Rules

- Implement `solve(x: torch.Tensor) -> torch.Tensor`, taking `max(0, x)` for each element.
- The output shape and dtype must match the input.
- The implementation must be vectorized; do not use a Python elementwise loop.

## Examples

```
Input: [-2.0, -0.5, 0.0, 0.5, 2.0]
Output: [0.0, 0.0, 0.0, 0.5, 2.0]
```

## Constraints

- The input is a float32 or float64 tensor of arbitrary shape
- No restriction on the value range (inputs may be all negative, all positive, or contain zeros)
