# GLU (Gated Linear Unit)

## Problem

A Gated Linear Unit splits the input in half along one dimension: one half acts as the "value" and the other goes through sigmoid to act as the "gate":

$$
\text{GLU}(x) = a \odot \sigma(b), \quad a, b = \text{split}_{dim}(x, 2)
$$

That is, `x` is split evenly along `dim` into the front half $a$ and the back half $b$; the output is $a \cdot \text{sigmoid}(b)$, halved along that dimension. Gated structures are widely used in Transformer FFNs (e.g. SwiGLU).

## Rules

- Implement `solve(x: torch.Tensor, dim: int = -1) -> torch.Tensor`.
- Split in half along `dim` with `torch.chunk(x, 2, dim)`; multiply the first half by `sigmoid(second half)`.
- The output length along `dim` is half the input length; use a vectorized implementation.
- An odd length along `dim` cannot be split evenly and must raise (`torch.chunk` already errors out, so no special handling is needed).

## Examples

```
Input: x = [1.0, 2.0, 3.0, 4.0], dim = -1
Split: a = [1.0, 2.0], b = [3.0, 4.0]
Output: [0.95257413, 1.96402752]   # 1*σ(3), 2*σ(4)
```

## Constraints

- Input is a float32/float64 tensor; the length of the `dim` axis must be even
- The result matches `torch.nn.functional.glu`
