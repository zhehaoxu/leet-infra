# Sigmoid

## Problem

Implement the elementwise Sigmoid activation, which compresses any real number into the $(0, 1)$ interval:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

The naive form overflows `exp(-x)` to `inf` when $x$ is a large negative number. The classic numerically stable trick branches on the sign:

- $x \ge 0$: $\sigma(x) = \frac{1}{1 + e^{-x}}$ (the exponent argument is negative, so it cannot overflow)
- $x < 0$: $\sigma(x) = \frac{e^{x}}{1 + e^{x}}$ (again keeping the exponent argument negative)

## Rules

- Implement `solve(x: torch.Tensor) -> torch.Tensor`, returning the elementwise sigmoid.
- You must use the **sign-branching stable form** above: for any input, intermediate computations must not produce `inf`/`NaN`.
- Output shape and dtype match the input; use a vectorized implementation.

## Example

```
Input: [-2.0, 0.0, 2.0]
Output: [0.11920, 0.5, 0.88080]
```

## Constraints

- Input is a float32 or float64 tensor of any shape
- Values may reach ±800 (far beyond the safe range of `exp` in float32)
