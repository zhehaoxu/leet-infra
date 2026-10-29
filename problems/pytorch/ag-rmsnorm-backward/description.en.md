# RMSNorm Backward: the Missing Mean-Subtraction Term

## Problem Description

LLaMA-style models replace LayerNorm with RMSNorm: divide by the root-mean-square only, no mean subtraction, so normalization collapses to one sum-of-squares reduction along the last axis. The backward is not simply the LayerNorm formula with \(\mu\) deleted — the \(\text{mean}(g)\) term does disappear, but \(\hat x\cdot\text{mean}(g\circ\hat x)\) survives, and \(\hat x\) is now \(x\) itself rather than a centered residual. Given `x, gamma, dY`, return `(dX, dGamma)`.

## Rules

- The full signature and parameter semantics live only in the starter docstring; that is the single source of truth for judging.
- Forward: \(rms=\sqrt{\text{mean}(x^2,\text{last})+eps}\), \(\hat x=x/rms\), \(y=\gamma\circ\hat x\).
- With \(g=dY\circ\gamma\): \(dX=\big(g-\hat x\cdot\text{mean}(g\circ\hat x)\big)/rms\) — there is **no** \(\text{mean}(g)\) term.
- \(d\gamma=\sum_{\text{leading}} dY\circ\hat x\), always of shape `(H,)`.
- `eps` may be 0.0 while a negative value raises ValueError; a wrong `gamma` length or `dY` shape raises ValueError.

## Example

One row of 4 features with `eps=0.0`, same inputs as the LayerNorm problem; numbers come from the EXAMPLE_* constants in test_cases.py:

```
x     = [[-1, 1, -1, 1]]   gamma = [2, 1, 2, 1]   dY = [[1, 2, 3, 4]]
rms = 1, xhat = x, g = dY*gamma = [2, 2, 6, 4], mean(g*xhat) = -0.5
dX     = [[1.5, 2.5, 5.5, 4.5]]     dGamma = [-1, 2, -3, 4]
```

Column 0: \(2-(-1)(-0.5)=1.5\). With identical inputs LayerNorm returns \(-2\); the gap is exactly the removed \(\text{mean}(g)=3.5\).

## Constraints

- Any leading axes and float32/float64 are supported; accumulate in float64 then cast back
- Cross-checked against `torch.autograd.grad` of a hand-built \(x.pow(2).mean(-1)\) forward, atol=1e-9 (float64) and 1e-5 (float32)
