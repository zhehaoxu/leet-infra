# Weight-Only INT8 Dequantized Matrix Multiplication

## Problem

The most commonly used quantization tier in LLM inference is **weight-only** (W8A16 / W4A16): only the weights
are stored as integers to save memory and bandwidth, while activations stay in floating point (accuracy barely
drops). The dequantization is a single scale vector **per output channel**. This problem simulates running one
matrix multiply of a `Linear` layer with int8 weights.

## Rules

Implement `solve(x, Wq, w_scale)`:

- `x`: float activations $[M, K]$ (not quantized).
- `Wq`: int8 weights $[N, K]$, where $N$ is the number of output channels (each `Wq[n]` is one row).
- `w_scale`: the symmetric per-output-channel scale $[N]$ (float, equal to that channel's $\text{absmax}/127$).

The true floating-point weight is $W[n,k] = \text{w\_scale}[n]\cdot Wq[n,k]$; compute $x$ through a fully
connected layer with it:

$$\text{out}[m,n] = \sum_{k} x[m,k]\,W[n,k] = \text{w\_scale}[n]\sum_{k} x[m,k]\,Wq[n,k]$$

Return `out` (float, shape `[M, N]`, with the same dtype as `x`).

> Key structure: the per-channel scale is independent of the summation index $k$, so it can be **pulled out of
> the matrix multiply** — first do `x @ Wqᵀ`, then multiply each column (output channel) by `w_scale`. This is
> exactly the basis for the hardware pattern of "accumulate in integer / half precision, then rescale
> uniformly", and it is also the way to avoid materializing the whole dequantized weight (saving memory). The
> floating-point results of the two formulations should agree within tolerance.

## Example

```
x       = [[1.0, 2.0, 0.0],
           [0.5, -1.0, 3.0]]
Wq      = [[10, -100, 50],
           [127,   0, -64]]
w_scale = [0.01, 0.02]

x @ Wqᵀ = [[-190, 127],      out = multiply each column by w_scale
            [ 255, -128.5]]
        = [[-1.9, 2.54],
           [ 2.55, -2.57]]
```

## Constraints

- `Wq` is int8 (may contain ±127/-128); accumulate in float (since `x` is float, `x @ Wq.float().T` suffices)
- The output dtype and shape are as above; the result should match
  `F.linear(x, Wq.float() * w_scale[:, None])` (the built-in oracle)
- Do not modify the inputs
