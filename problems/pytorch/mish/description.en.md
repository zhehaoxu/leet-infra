# Mish

## Problem

Mish is a self-gated smooth curve with a slight dip in the negative region (similar to Swish); the paper claims it outperforms ReLU on certain tasks:

$$
\text{Mish}(x) = x \cdot \tanh\!\big(\text{SoftPlus}(x)\big) = x \cdot \tanh\!\left(\log(1+e^{x})\right)
$$

## Rules

- Implement `solve(x: torch.Tensor) -> torch.Tensor` with the same shape and dtype as the input.
- Primitives such as `torch.tanh`, `torch.exp`, `torch.log1p` are allowed.
- **Numerical stability**: the inner $\text{SoftPlus}(x)=\log(1+e^x)$ must be computed stably with a sign branch (`exp(x)` overflows for large values, and precision suffers when $x$ is close to 0), otherwise Mish yields NaN. Recommended: `softplus = where(x>0, x + log1p(exp(-x)), log1p(exp(x)))`.
- Vectorized implementation.

## Examples

```
Input: [-1.0, 0.0, 1.0]
Output: [-0.30340147, 0.0, 0.86509842]
```

## Constraints

- The input is a float32 or float64 tensor of arbitrary shape
- For large $x$ the output $\to x$, and for very negative $x$ the output $\to 0$ (neither may produce NaN/inf)
