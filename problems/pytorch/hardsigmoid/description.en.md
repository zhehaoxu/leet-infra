# HardSigmoid

## Problem

HardSigmoid approximates the Sigmoid with a clipped linear function; its output always lies in $[0,1]$:

$$
\text{HardSigmoid}(x) = \frac{\text{clip}(x+3,\,0,\,6)}{6} =
\begin{cases} 0, & x \le -3 \\ (x+3)/6, & -3 < x < 3 \\ 1, & x \ge 3 \end{cases}
$$

It is the coefficient part of HardSwish (HardSwish $= x \cdot$ HardSigmoid).

## Rules

- Implement `solve(x: torch.Tensor) -> torch.Tensor`, with the same shape and dtype as the input.
- Vectorized implementation; the output must lie in $[0,1]$.

## Examples

```
Input: [-4.0, -3.0, 0.0, 3.0, 4.0]
Output: [0.0, 0.0, 0.5, 1.0, 1.0]
```

## Constraints

- The input is a float32 or float64 tensor of any shape
- On the ramp, at $x=-1$ the output is $(-1+3)/6 = 1/3$
