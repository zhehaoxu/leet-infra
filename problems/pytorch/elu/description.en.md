# ELU

## Problem

ELU (Exponential Linear Unit) saturates exponentially to $-\alpha$ in the negative region, giving a mean closer to 0 and faster convergence:

$$
\text{ELU}(x) =
\begin{cases} x, & x > 0 \\ \alpha\,(e^{x} - 1), & x \le 0 \end{cases}
$$

## Rules

- Implement `solve(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor`, with shape and dtype matching the input.
- Vectorized implementation; Python loops are not allowed.
- **Precision tip**: in the negative region, $e^x - 1$ suffers floating-point cancellation error when $x$ is close to 0; use `torch.expm1(x)` to compute $e^x-1$ directly for better accuracy.

## Examples

```
Input: x = [-1.0, 0.0, 1.0], alpha = 1.0
Output: [-0.63212055, 0.0, 1.0]     # expm1(-1) = e^-1 - 1
```

## Constraints

- Input is a float32 or float64 tensor of arbitrary shape
- `alpha` is a non-negative float scalar; when $x$ is very negative the output approaches $-\alpha$
