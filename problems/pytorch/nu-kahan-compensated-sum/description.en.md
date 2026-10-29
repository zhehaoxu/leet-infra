# Compensated and Pairwise Summation: Pushing the Accumulation Error Back

## Problem Description

Add up every element of a tensor in row-major order and return a Python float. Floating-point addition is not associative: once \(a+b\) is rounded inside `(a+b)+c`, the low bits of \(c\) are gone forever, and the error of a sequential accumulation grows linearly with the element count. Loss accumulation, gradient all-reduce, BatchNorm statistics, and the fp16 sum after loss scaling all do exactly this, so "how you add" is the first question numerical stability ever asks. This problem wants all four summation formats implemented with bit-reproducible output — the same data yields three different integers depending on the mode, and that difference is the point.

## Rules

- The full signature and parameter semantics live only in the starter docstring; that is the single source of truth for judging.
- Convert to float64 before accumulating, and never call `torch.sum` or `math.fsum`: all four modes are written step by step by hand.
- `naive`: start from \(s=0\) and do \(s \mathrel{+}= v\) in order.
- `kahan`: \(y=v-c,\ t=s+y,\ c=(t-s)-y,\ s=t\), returning only \(s\) and not the compensation \(c\).
- `kahan_babuska` (Neumaier): \(t=s+v\); if \(|s|\ge|v|\) then \(c \mathrel{+}= (s-t)+v\) else \(c \mathrel{+}= (v-t)+s\); return \(s+c\). It patches the plain Kahan flaw of discarding the compensation whenever the new element is larger than the running sum.
- `pairwise`: sum consecutive groups of `block` elements sequentially, then merge the block sums by recursive bisection. With `block >= numel` it is bitwise identical to `naive`.
- An empty tensor returns 0.0; if any ±inf/NaN appears in the flattened data, **every mode degrades to the sequential naive accumulation** (IEEE propagation), because the compensation terms would necessarily compute `inf - inf`.
- Integer/boolean tensors, an unknown `mode`, and `block < 1` all raise ValueError.

## Example

`x = [1e16, 1.0, -1e16, 1.0]`, whose exact sum is 2 (neither of the two 1.0 values should be swallowed):

```
x = [1e16, 1.0, -1e16, 1.0]

naive           -> 1.0
kahan           -> 1.0
kahan_babuska   -> 2.0
pairwise(b=8)   -> 1.0
pairwise(b=2)   -> 0.0
```

`1e16 + 1.0` is just `1e16` in double (ulp(1e16) = 2), so both 1.0 values vanish; plain Kahan computes `t = 1e16 + 1` = `1e16` with \(c=(t-s)-y=0\), and when \(-10^{16}\) arrives it takes the wrong branch since \(|s|=10^{16}\gg|v|=1\), losing the compensation again. Neumaier compares \(|s|\ge|v|\) and returns \(s+c\), so both 1.0 values survive inside \(c\). `pairwise` with `block=8` is one sequential pass over the whole array and therefore matches naive; with `block=2` the two block sums \((10^{16}+1)\approx10^{16}\) and \((-10^{16}+1)\approx-10^{16}\) cancel to exactly 0.0, which is worse than naive.

## Constraints

- float32/float64 tensors of any shape; empty tensors and non-finite values are valid inputs, not errors
- the oracle is `math.fsum` (exhaustively carried exact summation) plus hand-computed integers, with a 1e-12 relative tolerance on well-conditioned float64 data
- time \(O(n)\); `pairwise` adds \(O(n/\text{block})\) block sums and one bisection merge
