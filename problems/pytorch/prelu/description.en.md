# PReLU (Learnable Negative Slope)

## Problem

PReLU turns the negative slope of Leaky ReLU into a **learnable parameter**, which is usually shared per channel:

$$
\text{PReLU}(x) =
\begin{cases} x, & x > 0 \\ w \cdot x, & x \le 0 \end{cases}
$$

where $w$ is a tensor aligned with the number of channels.

## Rules

- Implement `solve(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor`.
- `weight` is either a scalar tensor or a 1-D tensor of length **C (number of channels)** (by PyTorch convention it acts on the `dim=1` channel dim and must be reshaped to `(1, C, 1, 1, ...)` before broadcasting).
- The output shape and dtype match `x`; implement it vectorized.
- `weight=0` degenerates to ReLU.

## Examples

```
Input: x = [-2.0, -1.0, 0.0, 1.0, 2.0], weight = 0.25 (scalar)
Output: [-0.5, -0.25, 0.0, 1.0, 2.0]
```

Per-channel example: `x = [[-2, -4, 3], [-1, -5, 6]]`, `weight = [0.1, 0.5, 0.9]` (channel dim has length 3)
→ negative elements are scaled by their own channel: `[[-0.2, -2.0, 3.0], [-0.1, -2.5, 6.0]]`

## Constraints

- `x` is float32/float64 with shape `(N, C)` or `(N, C, ...)`
- The length of a 1-D `weight` must equal the channel count of `x` at `dim=1`
