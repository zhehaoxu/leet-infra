# LayerNorm Forward

## Problem

Layer Normalization normalizes the **trailing feature dimensions of each sample independently**, independent of the batch, with identical training and inference behavior:

$$
y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}\,\gamma + \beta,\qquad
\mu,\sigma^2 = \text{mean/var over the last } \dim(\gamma) \text{ dims}
$$

Every Transformer block relies on it to stabilize the input distribution of each layer.

## Rules

- Implement `solve(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor`.
- The normalized dimensions are the **trailing** `weight.dim()` dimensions of x (`weight.shape` is the normalized_shape and must match the trailing dimensions of x).
- The variance uses the **biased** convention (divide by $n$, consistent with `F.layer_norm`) — LayerNorm has no BatchNorm-style running stats and applies no Bessel correction.
- $\epsilon$ is added **inside the sqrt** (the denominator is $\sqrt{\sigma^2+\epsilon}$, not $\sqrt{\sigma^2}+\epsilon$).
- Compute the statistics with `keepdim` and broadcast; use a vectorized implementation.

## Examples

```
Input: x = [[1.0, 2.0, 3.0, 4.0]], γ = [1,1,1,1], β = [0,0,0,0], eps = 1e-5
μ = 2.5, σ²(biased) = 1.25
Output: [[-1.34163547, -0.44721183, 0.44721183, 1.34163547]]
```

## Constraints

- float32/float64, arbitrary shape (number of dims ≥ weight.dim())
- A constant vector (σ²=0) must not produce NaN; the result must match `torch.nn.functional.layer_norm`
