# GELU (Exact and tanh Approximation)

## Problem

Implement the GELU (Gaussian Error Linear Unit) activation function—the standard activation of the Transformer family (BERT, GPT). It weights the input by the CDF of the standard normal distribution:

$$
\text{GELU}(x) = x \cdot \Phi(x) = \frac{x}{2}\left(1 + \text{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right)
$$

In engineering there is also a cheaper tanh approximation (commonly used in the BERT/LLaMA papers):

$$
\text{GELU}_{\tanh}(x) = \frac{x}{2}\left(1 + \tanh\!\left(\sqrt{\frac{2}{\pi}}\,(x + 0.044715\,x^3)\right)\right)
$$

## Rules

- Implement `solve(x: torch.Tensor, approximate: str = "none") -> torch.Tensor`.
- `approximate == "none"` uses the exact erf formula; `approximate == "tanh"` uses the tanh approximation above.
- Any other value of `approximate` raises `ValueError` (consistent with `torch.nn.functional.gelu`).
- `torch.erf` / `torch.tanh` may be used as primitives; vectorized implementation.

## Examples

```
Input: [-1.0, 0.0, 1.0]  (approximate="none")
Output: [-0.15865, 0.0, 0.84134]
```

## Constraints

- Input is a float32 or float64 tensor of arbitrary shape
- The results of both modes must match the corresponding mode of `torch.nn.functional.gelu` within `1e-5`
