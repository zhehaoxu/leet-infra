# matmul/linear Backward: dX and dW with the Transpose and Broadcast Traps

## Problem Description

`Y = X @ W.T` is exactly the forward of `nn.Linear`, and its backward is the most frequently invoked kernel in any framework. There are only two rules: dX multiplies the upstream gradient by the weight once more, while dW transposes the upstream gradient and multiplies by the input. What is hard is the **dimension bookkeeping**: with several leading batch axes, dW must fold all of them into one sum, and writing a single \((M,N)^T @ (M,K)\) silently drops that accumulation. Given `dY`, `X` and `W`, return `(dX, dW)`.

## Rules

- The full signature and parameter semantics live only in the starter docstring; that is the single source of truth for judging.
- `dX = dY @ W` with shape `(..., M, K)`; `dW = Σ_b dY_b.T @ X_b` with shape `(N, K)`, identical to `W`.
- The leading axes of `dY` and `X` must match exactly (no implicit broadcasting allowed) and `W` must be strictly 2-D.
- `M = 1` or zero-sized axes must degrade through the same formula rather than returning zeros as a special case.
- Accumulate in float64 internally; the dtypes of `dX`/`dW` follow `X`/`W`. Any shape mismatch raises ValueError.

## Example

`X:(2,2)`, `W:(3,2)`, `dY:(2,3)` all ones; numbers come from the EXAMPLE_* constants in test_cases.py:

```
dY = [[1, 1, 1],      X = [[1, 2],      W = [[1, 0],
      [1, 1, 1]]          [3, 4]]          [2, 1],
                                          [0, 3]]
dX = dY @ W          = [[3, 4],          dW = dY.T @ X = [[4, 6],
                                            [3, 4]]                      [4, 6],
                                                                         [4, 6]]
```

Each entry of `dX` sums over the three rows of `W` (\(1+2+0=3\), \(0+1+3=4\)); each row of `dW` is \(1\cdot1+1\cdot3=4\) and \(1\cdot2+1\cdot4=6\).

## Constraints

- Any number of leading axes (3-D and 4-D inputs) is supported; cross-checked against `torch.autograd.grad((X @ W.T), (X, W))`
- atol=1e-10 for float64 and 1e-5 for float32; the batch folding of `dW` must match autograd entry by entry
