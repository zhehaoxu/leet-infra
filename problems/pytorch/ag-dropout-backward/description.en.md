# Dropout Backward: Mask Gating and the Train/Eval Asymmetry

## Problem Description

Inverted dropout rescales in the forward (\(y=x\cdot mask/(1-p)\)), and differentiating that w.r.t. \(x\) reproduces the same factor, giving \(dx=dY\cdot mask/(1-p)\). That sounds like one line, but the three real traps are all conventions: the division by \(1-p\) happens **only in training mode**; with `train=False` the forward is the identity so the backward must return `dY` unchanged (ignoring both `mask` and `p`); and `mask` may be a **broadcastable** coarse mask (`dropout2d` uses shapes like `(1,C,1,1)` shared over whole channels). Given `dY, mask, p, train`, return `dx`.

## Rules

- The full signature and parameter semantics live only in the starter docstring; that is the single source of truth for judging.
- `train=True`: \(dx=dY\cdot mask/(1-p)\) — dropped positions get zero, kept positions are amplified by \(1/(1-p)\).
- `train=False`: return a copy of `dY`, never touching `mask`, never dividing by \((1-p)\), and never validating `p`.
- `mask` must have the shape of `dY` or be broadcastable to it; in training mode a non-broadcastable mask raises ValueError.
- `p` must lie in \([0,1)\): `p=1` would make \(1-p=0\) and raises ValueError like a negative value does.

## Example

`p=0.5` (a factor of 2); numbers come from the EXAMPLE_* constants in test_cases.py:

```
dY   = [[1, 2],     mask = [[1, 0],     dx = [[2, 0],
       [3, 4]]              [0, 1]]            [0, 8]]
```

With \(dx=dY\cdot mask/0.5\): position (0,0) gives \(1/0.5=2\), (1,1) gives \(4/0.5=8\), and the two dropped cells are 0.

## Constraints

- `mask` may be a bool, int or float tensor; arithmetic runs in float64 and the result has the dtype of `dY`
- Cross-checked against `F.dropout` plus `torch.autograd.grad` under the same mask, with \(E[mask]/(1-p)=1\) verifying the expectation is preserved
