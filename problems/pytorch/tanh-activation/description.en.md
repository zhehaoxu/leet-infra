# Tanh

## Problem

Implement the elementwise hyperbolic-tangent activation:

$$
\tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}} \in (-1, 1)
$$

It was the default activation in early neural networks and is a good example for understanding "numerical stability": using $e^{x}$ directly overflows on large positive numbers.

## Rules

- Implement `solve(x: torch.Tensor) -> torch.Tensor`; output shape and dtype match the input.
- The implementation must be vectorized — no per-element Python loops.
- **Numerical stability**: the naive `(exp(x)-exp(-x))/(exp(x)+exp(-x))` overflows `exp(800)` to `inf`, giving `inf/inf = NaN`. Use the identity $\tanh(x)=\frac{1-e^{-2|x|}}{1+e^{-2|x|}}$ and branch on the sign; $e^{-2|x|}$ is always $\le 1$ and cannot overflow.

## Example

```
Input: [-2.0, -1.0, 0.0, 1.0, 2.0]
Output: [-0.96402758, -0.76159412, 0.0, 0.76159412, 0.96402758]
```

## Constraints

- Input is a float32 or float64 tensor of any shape
- No bound on the values (extremes such as ±800 may appear; they must saturate to ±1 rather than become NaN)
