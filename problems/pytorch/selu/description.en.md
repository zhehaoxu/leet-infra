# SELU Self-Normalizing Activation (SELU)

## Problem

SELU (Scaled Exponential Linear Unit) is an ELU variant with a fixed scaling. What makes it special is not the shape of the curve but the two **magic constants** $\lambda$ (scale) and $\alpha$: as long as the weights are initialized with the matching scheme, a stack of SELU layers automatically pulls every layer's activations back to mean 0 and variance 1 — i.e. "self-normalization", with no extra BatchNorm/LayerNorm.

## Rules

- Implement `solve(x)`, returning elementwise:
  $$ \operatorname{SELU}(x)=\begin{cases}\lambda\,x & x>0\\ \lambda\,\alpha\,(e^{x}-1) & x\le 0\end{cases} $$
- Constants: $\alpha=1.6732632423543772848170425930144$, $\lambda=1.0507009873554804977367086399070$.
- Shape-preserving and dtype-preserving; the result must match `F.selu(x)`.

## Examples

```
Input: x = [-2.0, -0.5, 0.0, 0.5, 2.0, 7.0]
Output: [-1.52017, -0.69176, 0.0, 0.52535, 2.1014, 7.35491]
(For x>0 the value is scaled up by about 1.05×; for x<=0 it is a negative saturating exponential, continuous at 0)
```

## Constraints

- Supports float32 / float64 inputs; output dtype matches the input
- $\lambda,\alpha$ must use the high-precision values given in the statement, otherwise the result deviates from `F.selu`
