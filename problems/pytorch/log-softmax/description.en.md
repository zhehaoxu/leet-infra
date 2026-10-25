# Numerically Stable Log-Softmax

## Problem

log-softmax applies softmax and then takes the logarithm: $\log\text{softmax}(x)_i = x_i - \log\sum_j e^{x_j}$.
Classification losses (cross entropy = `log_softmax` + `NLL`) use it almost everywhere. The naive
`log(softmax(x))` overflows `exp` to `inf` for large logits and underflows for small ones, giving `log(0)=-inf`.
Implement the stable version with the **subtract-maximum** trick.

## Rules

- Implement `solve(x, dim=-1)`, computing log-softmax along `dim` with the same shape as the input.
- Stable formula:
  $$ \log\text{softmax}(x)_i = (x_i - m) - \log\!\sum_j e^{x_j - m},\quad m=\max_j x_j $$
- Subtracting the maximum does not change the result (numerator and denominator both multiply by $e^{-m}$ and cancel), but it squeezes the exponent arguments to $\le 0$, preventing overflow.
- The result must match `F.log_softmax(x, dim=dim)`.

## Example

```
Input: x = [[1, 2, 3],
          [3, 1, 0]] (along the last dim)
Output: [[-2.4076, -1.4076, -0.4076],
      [-0.1698, -2.1698, -3.1698]]
(Exponentiating each row rebuilds that row's softmax probabilities, which sum to 1)
```

## Constraints

- float32 input; even with logits in the thousands, no `inf/nan` may appear
- `exp(output)` summed along `dim` must equal 1
