# Activation Backward Suite: Elementwise Derivatives and Subgradient at Zero

## Problem Description

An activation backward looks like reciting eight formulas, but the real difficulty lives at the **kinks and the non-elementary cases**: `relu` must take 0 at \(x=0\) (PyTorch's subgradient convention — 0.5 is wrong), `elu` is differentiable there with value 1, `gelu` needs the exact \(\Phi(x)+x\phi(x)\) rather than the tanh approximation, and `silu` is \(\sigma+x\sigma(1-\sigma)\). Given `x, dY, kind`, return the elementwise `dx`.

## Rules

- The full signature and parameter semantics live only in the starter docstring; that is the single source of truth for judging.
- `kind` is one of `'relu'`, `'leaky_relu'`, `'elu'`, `'sigmoid'`, `'tanh'`, `'gelu'`, `'silu'`, `'softplus'`; anything else raises ValueError.
- Formulas: relu \(=\mathbb 1[x>0]\); leaky_relu \(=1\) or `negative_slope`; elu \(=1\) or \(e^x\); sigmoid \(=s(1-s)\); tanh \(=1-t^2\); softplus \(=\sigma(x)\); gelu \(=\Phi(x)+x\phi(x)\); silu \(=\sigma(x)+x\sigma(x)(1-\sigma(x))\).
- Values at \(x=0\): relu 0, leaky_relu `negative_slope`, elu 1, silu 0.5, softplus 0.5.
- `negative_slope` is used only by `leaky_relu`; other kinds must ignore it without erroring. A shape mismatch between `x` and `dY` raises ValueError.

## Example

`kind='relu'` with all-ones `dY`; numbers come from the EXAMPLE_* constants in test_cases.py:

```
x  = [-2.0, -0.5, 0.0, 0.5, 2.0]
dY = [ 1.0,  1.0, 1.0, 1.0, 1.0]
dx = [ 0.0,  0.0, 0.0, 1.0, 1.0]      # x == 0 gives 0, not 0.5
```

## Constraints

- `x` may have any shape (including 0-dim tensors); compute in float64 and return the dtype of `x`
- Compared entry by entry with `torch.autograd.grad` of the matching `F.*`, atol=1e-9 in float64; `gelu` uses the exact erf form, which differs from the tanh approximation by about \(10^{-3}\) near \(|x|\approx 2\)
