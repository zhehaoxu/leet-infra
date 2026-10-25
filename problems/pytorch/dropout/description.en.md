# Dropout (train / eval)

## Problem

Inverted Dropout: during training, randomly set elements to 0 with probability $p$, and scale the **retained** elements by dividing by $(1-p)$, so that the output expectation equals the input. At inference time nothing changes (identity).

$$ y_i = \frac{m_i}{1-p}\,x_i,\qquad m_i \sim \text{Bernoulli}(1-p) $$

## Rules

- Implement `solve(x, p=0.5, training=True, seed=1234)`.
- When `training=False` or `p <= 0`, return the **identity** (the original values; even eval with `p=0.9` must be identity).
- During training the output can only be $0$ or $x/(1-p)$ (retained values are scaled exactly by $1/(1-p)$, keeping the expectation unchanged).
- **The following random-number conventions must be followed exactly** (otherwise you will not match the reference implementation):
  1. Use a local generator: `g = torch.Generator(device=x.device).manual_seed(seed)`;
  2. Mask: `mask = torch.rand(x.shape, generator=g, device=x.device, dtype=x.dtype) >= p` (keep positions where the value is `>= p`);
  3. Output: `x * mask / (1 - p)`.
- **Must not consume the global RNG**: only the local `Generator` above may be used; the tests verify with a fixed baseline sequence that the global random state is untouched.
- The same `seed` must produce the same mask; different `seed`s must produce different masks. The output dtype follows the input (float64 supported).

## Examples

```
Input: x=ones(1,8), p=0.5, training=True, seed=1234
   generate the mask per the Generator convention above; only position 5 is retained
Output: [[0,0,0,0,0,2,0,0]]    # retained value = 1/(1-0.5) = 2.0
```

## Constraints

- float32/float64 inputs
- `0 <= p < 1`; the result is fully determined by `seed` (reproducible)
