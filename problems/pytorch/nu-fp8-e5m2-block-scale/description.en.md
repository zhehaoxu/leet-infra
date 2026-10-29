# FP8 E5M2 Block Scaling: Power-of-Two Shared Scale and Grid Quantization

## Problem Description

FP8 keeps only 3 significant bits (E5M2 is 1 sign + 5 exponent + 2 explicit mantissa bits), so its raw range is many orders of magnitude narrower than bf16 and it is only usable with **a scale shared per block**: Microscaling, DeepSeek's tile-wise scaling and block-quantized KV cache all mean "one power-of-two exponent per block, FP8 mantissas inside it". This problem runs the whole chain: cut the tensor into consecutive groups of `block` elements, give each block the **smallest power-of-two scale that cannot overflow**, round the block entries onto the E5M2 grid (ties-to-even), then multiply the scale back. The output stays a float type so it can be compared bitwise.

## Rules

- The full signature and parameter semantics live only in the starter docstring; that is the single source of truth for judging.
- No `x.to(torch.float8_e5m2)` / `float8_e4m3fn` and no autocast — you must enumerate the E5M2 grid yourself.
- E5M2 magnitudes: normals \((1+M/4)\cdot2^e\) with \(e=-14..15\), \(M=0..3\), largest finite value \(57344=1.75\cdot2^{15}\); subnormals \(M\cdot2^{-16}\) (step \(2^{-16}\approx1.526\times10^{-5}\)). Only four grid points per octave.
- Procedure (flatten row-major; the last block may be shorter than `block`):
  1. \(m=\) the maximum \(|x|\) inside the block;
  2. \(\text{scale}=2^e\) where \(e\) is **the smallest integer with \(m\le57344\cdot2^e\)**, clamped to \([-127,127]\); if \(m=0\) then \(e=0\);
  3. for every element, round \(t=x/\text{scale}\) to the nearest E5M2 grid point (on a tie pick the one whose mantissa field \(M\) is even) and multiply the scale back; every zero in the output is written as +0.0.
- The scale guarantees \(|t|\le57344\), so quantization cannot overflow and the output never contains inf.
- If the block absmax is inf, or the block contains a NaN (or both) → **the whole block outputs NaN**; an all-zero block → all zeros.
- `block` must be an integer \(\ge1\); a non-float tensor, a non-tensor, or a bad `block` raises ValueError.

## Example

```
x = [1.0, 3.0, 5.0, 0.6, 0.5625]

block=5  absmax=5  scale=2^-13  ->  [1.0, 3.0, 5.0, 0.625, 0.5]
block=3  [1,3,5] [0.6,0.5625]   ->  [1.0, 3.0, 5.0, 0.625, 0.5]

x=[4608.0]  block=1  ->  4096.0
x=[4915.2]  block=1  ->  5120.0
```

With `block=5` the absmax is 5, the smallest \(e\) with \(5\le57344\cdot2^e\) is \(e=-13\), so multiplying by \(2^{13}=8192\) brings the block onto the grid: 1, 3 and 5 are grid values to begin with, 0.6 lands on 0.625 and 0.5625 on 0.5. `block=3` splits into two blocks whose scales are \(2^{-13}\) and \(2^{-16}\); since they differ by a power of two the grid sets nest and the rounding results are identical bitwise. `4608` sits exactly between the neighbors 4096(\(M=0\)) and 5120(\(M=1\)) of \([4096,8192)\), so ties-to-even keeps 4096; 4915.2 is closer to 5120 and rounds up.

## Constraints

- float32/float64 input of any shape, output of the same shape and dtype (computed internally in float64)
- the oracle is structurally different from the reference: enumerate every E5M2 magnitude (4 subnormals + \(30\times4\) normals) together with its bit pattern, find the nearest with `bisect`, break ties by pattern parity, and get the scale by a loop search instead of `log2`
- boundary cases must match bitwise: when the absmax is exactly \(57344\cdot2^k\) the scale has to be \(2^k\) (one step smaller halves the largest value)
