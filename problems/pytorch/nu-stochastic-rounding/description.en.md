# Stochastic Rounding: Unbiased SR and the fwd/bwd Biased Grid Choice

## Problem Description

The nastiest part of low-precision accumulation is not that the error is large but that it is **directed**: truncation (RZ) always pushes the magnitude toward zero, so after thousands of gradient additions the weights shrink systematically. Stochastic rounding replaces the bias with "roll a dice in proportion to the residual", driving the expected bias to zero at the price of per-step noise, which is why a Philox/Threefry noise stream is needed in practice. Here the noise is passed in as an argument so the judge is fully deterministic: given \(x\) and uniform draws \(u\), decide element by element which end of the bf16 grid to land on, supporting unbiased SR plus the two biased variants "forward toward zero / backward away from zero".

## Rules

- The full signature and parameter semantics live only in the starter docstring; that is the single source of truth for judging.
- No `x.to(torch.bfloat16)`; write the bit manipulation yourself.
- The key structure: **bf16 is the high 16 bits of a float32, so grid points are uniform in the "high 16-bit integer" space**. Read the 32-bit pattern of \(x\) as an unsigned integer \(u32\):

```
lo  = (u32 >> 16) << 16              low neighbor toward zero
hi  = ((u32 >> 16) + 1) << 16        next grid point away from zero
r   = (u32 & 0xFFFF) / 65536.0       position inside the step, 0 <= r < 1
out = hi  if u < p(r)  else  lo      strict less-than
```

- Probability of taking \(hi\) per variant: `sr` uses \(p=r\) (unbiased); `fwd` uses \(p=0\) when \(r\le0.5\) and \(p=2r-1\) otherwise (toward zero, for activations); `bwd` uses \(p=1\) when \(r\ge0.5\) and \(p=2r\) otherwise (away from zero, for gradients).
- Hence \(\mathbb E[\,|SR(x)|\,]=|x|\), \(\mathbb E[|fwd(x)|]\le|x|\) and \(\mathbb E[|bwd(x)|]\ge|x|\).
- Special values (none of them consult \(u\)): \(r=0\) (already on the grid, including ±0 and ±inf) returns the input unchanged and keeps the sign of ±0; NaN stays NaN; when \(|x|\) exceeds the largest finite bf16 value, taking \(hi\) overflows naturally to ±inf (no saturation).
- Validation: \(x\) and \(u\) must have the same shape and both be float32 (no broadcasting), every element of \(u\) must lie in \([0,1)\), and `variant` must be one of the three, otherwise ValueError.

## Example

Inside \([1,2)\) the bf16 ulp is \(2^{-7}\), so \(1+2^{-9}\) has \(r=0.25\) and \(1+3\cdot2^{-9}\) has \(r=0.75\):

```
x  = [1.001953125, 1.005859375]        r = [0.25, 0.75]
u  = [0.25, 0.5]

sr   -> [1.0, 1.0078125]
fwd  -> [1.0, 1.0]
bwd  -> [1.0078125, 1.0078125]

u2 = [0.0, 0.9]

sr   -> [1.0078125, 1.0]
fwd  -> [1.0, 1.0]
bwd  -> [1.0078125, 1.0078125]
```

For the first element under `sr`, \(p=r=0.25\) and the test is **strictly less than**, so \(u=0.25\) does not carry; with \(u=0\) it does. The second element has \(r=0.75\): `fwd` gives \(p=2(0.75)-1=0.5\), so \(u=0.5\) again sits exactly on the boundary and does not carry, while `bwd` has \(p=1\) whenever \(r\ge0.5\), and \(u<1\) always holds so it must carry.

## Constraints

- Element-wise, float32 tensors of any shape, output is float32 of the same shape
- the oracle runs the same bit manipulation on plain Python scalars via `struct` (never touching tensor views), compared bitwise
- unbiasedness is checked separately on 40000 samples by the sign of the mean error: `sr` must stay under 2e-4 in absolute bias, `fwd` must be clearly negative, `bwd` clearly positive, and \(E[|fwd|]<E[|sr|]<E[|bwd|]\)
