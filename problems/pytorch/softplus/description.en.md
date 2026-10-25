# SoftPlus

## Problem

SoftPlus is the smooth version of ReLU and a building block of activations such as GELU and Mish:

$$
\text{SoftPlus}(x) = \frac{1}{\beta}\log\!\left(1 + e^{\beta x}\right)
$$

It approaches $x$ when $\beta x$ is large, and approaches $\frac{1}{\beta}e^{\beta x}$ (tending to 0) when $\beta x$ is small.

## Rules

- Implement `solve(x: torch.Tensor, beta: float = 1.0, threshold: float = 20.0) -> torch.Tensor`.
- **Numerical stability**: `exp(beta*x)` overflows when `beta*x` is large. Follow the PyTorch convention: when `beta*x > threshold` return `x` directly (in that region softplus ≈ x), otherwise use `log1p(exp(beta*x))/beta`. Use `log1p` instead of `log(1+·)` to keep accuracy near zero.
- Vectorized implementation.

## Example

```
Input: x = [-2.0, 0.0, 2.0], beta = 1.0, threshold = 20.0
Output: [0.12692802, 0.69314718, 2.12692809]     # softplus(0)=ln2≈0.6931
```

## Constraints

- Input is a float32 or float64 tensor of any shape
- `beta > 0`; values may make `beta*x` reach the hundreds, so the linear branch is required to avoid overflow
