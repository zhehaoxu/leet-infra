# weight_norm Reparameterization Backward: Disambiguating dg and dv by Projection

## Problem Description

`weight_norm` writes the weight as \(W=g\cdot V/\lVert V\rVert\) slice by slice along axis 0. The payoff is a decoupling of direction and magnitude; the price is that \(g\) and \(\lVert v\rVert\) are **unidentifiable** — rescaling \(v_i\) by any \(c\) while keeping \(g_i\) leaves \(W\) untouched. The backward must therefore **project away** the component of \(dW\) along the unit direction \(\hat v\) before handing anything to \(dV\), otherwise that degree of freedom is counted once by \(dg\) and again by \(dV\). Given `V, g, dW`, return `(dg, dV)`.

## Rules

- The full signature and parameter semantics live only in the starter docstring; that is the single source of truth for judging.
- With \(\hat v_i=v_i/\lVert v_i\rVert_2\): \(dg_i=\sum_j dW_{i,j}\hat v_{i,j}\), the projection of \(dW\) onto the unit direction of that slice.
- \(dV_i=(g_i/\lVert v_i\rVert)\big(dW_i-\hat v_i\,dg_i\big)\) — the component along \(\hat v\) **must be removed first**.
- Two hard self-checks: \(\sum_j dV_{i,j}v_{i,j}=0\) (\(dV\) orthogonal to \(v\)), and rescaling slice \(i\) of \(V\) by any \(c\ne0\) (with \(g,dW\) fixed) leaves both \(W\) and \(dg\) unchanged.
- `V` must be at least 2-D, `g` must match axis 0, `dW` must match `V`; a zero-norm slice (direction undefined) raises ValueError.

## Example

Two 2-D slices with \(dW=I\); numbers come from the EXAMPLE_* constants in test_cases.py:

```
V = [[3, 4],     g = [2, 1]      dW = [[1, 0],
     [0, 5]]                            [0, 1]]
‖v‖ = [5, 5],  v̂ = [[0.6, 0.8], [0, 1]]
dg = [0.6, 1.0]                       # dg_i = dW_i · v̂_i
dV = [[0.256, -0.192], [0, 0]]        # (g/‖v‖)·(dW − v̂·dg)
```

Slice 0: \(0.4\cdot([1,0]-0.6[0.6,0.8])=0.4[0.64,-0.48]\); in slice 1 \(dW\) is collinear with \(\hat v\), so the subtraction cancels it exactly.

## Constraints

- Compute in float64; `dg` takes the dtype of `g` and `dV` that of `V`; `V` may be `(out, in)` or `(out, in, kh, kw)`
- Cross-checked against `torch.autograd.grad` of the equivalent forward `W = g·V/‖V‖` built as `torch.nn.utils.weight_norm` does, atol=1e-9 in float64
