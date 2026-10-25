# HardSwish

## Problem

HardSwish is a piecewise-linear approximation of Swish: it drops the expensive sigmoid and replaces it with a clipped linear function, which is common on mobile devices:

$$
\text{HardSwish}(x) = x \cdot \frac{\text{clip}(x+3,\,0,\,6)}{6} =
\begin{cases} 0, & x \le -3 \\ x(x+3)/6, & -3 < x < 3 \\ x, & x \ge 3 \end{cases}
$$

## Rules

- Implement `solve(x: torch.Tensor) -> torch.Tensor`, with the same shape and dtype as the input.
- Vectorized implementation (`clamp`/`clip` to $[0,6]$, then multiply by $x/6$).

## Examples

```
Input: [-4.0, -3.0, 0.0, 3.0, 4.0]
Output: [0.0, 0.0, 0.0, 3.0, 4.0]
```

## Constraints

- The input is a float32 or float64 tensor of any shape
- For $x \le -3$ the output is 0, and for $x \ge 3$ the output is $x$ (the plateau regions on both sides)
