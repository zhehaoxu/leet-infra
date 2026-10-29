# conv2d Backward: dx with the Flipped Kernel and dw as a Cross-Correlation

## Problem Description

The convolution backward is a pair of dual operations: \(dx\) scatters the upstream gradient back onto the input and is equivalent to a full convolution with the **kernel rotated by 180°**, while \(dw\) pairs input patches with upstream gradients inside each window, which is an ordinary cross-correlation. Both share the same `stride`/`padding` indexing yet need separate boundary handling — \(dx\) discards contributions that fall outside the input, \(dw\) folds away the batch and spatial axes. Given `dY, x, weight`, return `(dx, dw)`.

## Rules

- The full signature and parameter semantics live only in the starter docstring; that is the single source of truth for judging.
- Forward: \(Y[b,co,u,v]=\sum_{ci,k,l}x[b,ci,u\cdot s+k-p,\,v\cdot s+l-p]\cdot weight[co,ci,k,l]\).
- \(dw[co,ci,k,l]=\sum_b\sum_{u,v}dY[b,co,u,v]\cdot x[\dots]\), summed only over \((u,v)\) whose patch lies inside \(x\); the shape always equals `weight`.
- \(dx\) is the correlation of `dY` with the flipped kernel; positions that only exist because of padding are **not** written into `dx`, whose shape always equals `x`.
- Only `groups=1` is supported; `stride<1`, `padding<0`, a `Cin` mismatch and \(H_{out}\ne(H_{in}+2p-k_h)/s+1\) all raise ValueError.

## Example

A \(3\times3\) input, a \(2\times2\) all-ones kernel, `stride=1`, `padding=0`; numbers come from the EXAMPLE_* constants in test_cases.py:

```
x     = [[1, 2, 3],     weight = [[1, 1],     dY = [[1, 0],
       [4, 5, 6],                [1, 1]]           [0, 1]]
      [7, 8, 9]]
dx    = [[1, 1, 0],     dw = [[6,  8],
        [1, 2, 1],            [12, 14]]
        [0, 1, 1]]
```

\(dw[1,1]=x_{1,1}+x_{2,2}=5+9=14\); \(dx_{1,1}\) is covered by four output windows, two of which carry a nonzero \(dY\), giving 2.

## Constraints

- Accumulate in float64; the dtypes of `dx`/`dw` follow `x`/`weight`
- Cross-checked against `F.conv2d` plus `torch.autograd.grad` with the same `stride`/`padding`, atol=1e-9 in float64
