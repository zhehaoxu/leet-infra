# LayerNorm Backward: the Two Subtractions in dX and the γ/β Fold

## Problem Description

LayerNorm runs twice per Transformer block, and its backward is the easiest normalization derivative to get wrong: the forward couples two paths — subtracting the mean and dividing by the standard deviation — so dX must subtract **two** terms, \(\text{mean}(g)\) and \(\hat x\cdot\text{mean}(g\circ\hat x)\). Dropping either one still passes a subset of tests yet fails the autograd cross-check. Given `x, gamma, beta, dY`, return `(dX, dGamma, dBeta)`.

## Rules

- The full signature and parameter semantics live only in the starter docstring; that is the single source of truth for judging.
- Normalization runs along the **last axis** of `x`, with the biased variance (`unbiased=False`) and \(\hat x=(x-\mu)/\sqrt{var+eps}\).
- With \(g=dY\circ\gamma\): \(dX=\big(g-\text{mean}(g)-\hat x\cdot\text{mean}(g\circ\hat x)\big)/\sigma\), both means along the last axis.
- \(d\gamma=\sum_{\text{leading}} dY\circ\hat x\) and \(d\beta=\sum_{\text{leading}} dY\), always of shape `(H,)`, even for 1-D `x`.
- `beta` enters no intermediate value: perturbing it must leave all three outputs bit-identical. Mismatched shapes or an illegal `eps` raise ValueError.

## Example

One row of 4 features with `eps=0.0`; numbers come from the EXAMPLE_* constants in test_cases.py:

```
x      = [[-1, 1, -1, 1]]     gamma = [2, 1, 2, 1]   beta = [0, 0, 0, 0]
dY     = [[ 1, 2,  3, 4]]
mu = 0, var = 1, xhat = x,  g = dY*gamma = [2, 2, 6, 4]
mean(g) = 3.5,  mean(g*xhat) = -0.5
dX     = [[-2, -1, 2, 1]]     dGamma = [-1, 2, -3, 4]   dBeta = [1, 2, 3, 4]
```

Column 0: \(2-3.5-(-1)(-0.5)=-2\); column 2: \(6-3.5-(-1)(-0.5)=2\).

## Constraints

- Any leading axes (`(H,)`, `(B,H)`, `(B,T,H)`) and float32/float64 are supported; accumulate in float64 then cast back
- Cross-checked against `torch.autograd.grad` of `torch.nn.functional.layer_norm` with atol=1e-9 (float64) and 1e-5 (float32)
