# bfloat16 Rounding: Bitwise RNE, Truncation and Ties-to-Even

## Problem Description

bf16 has no dedicated hardware conversion instruction: it *is* the high 16 bits of a float32 — the mantissa loses 16 bits while the exponent and sign are kept verbatim. Every implementation of "push the weights into bf16", "all-reduce gradients in bf16" or "simulate mixed precision on the CPU" eventually has to answer the same question: **how do you round the 16 discarded bits**. This problem asks you to implement three rounding modes directly on the float32 bit pattern (round-to-nearest-even, round-toward-zero, round-to-nearest-with-ties-away-from-zero), return the result still as float32 (a fake cast), and match the hardware conversion bit for bit.

## Rules

- The full signature and parameter semantics live only in the starter docstring; that is the single source of truth for judging.
- No `x.to(torch.bfloat16)` / `x.bfloat16()`, and no borrowing numpy or struct for the type conversion.
- Let `u` be the 32-bit pattern read as an **unsigned** integer, \(L=u\,\&\texttt{0xFFFF}\) the discarded low part, and bit 16 (the lowest bit of `u >> 16`) the retained last mantissa bit:

```
rz  : (u & 0xFFFF0000)                      -> take the high 16 bits
rne : u + 0x7FFF + ((u >> 16) & 1)           -> then the high 16 bits
rna : u + 0x8000                             -> then the high 16 bits
```

- For rne: carry when \(L>\texttt{0x8000}\); when \(L=\texttt{0x8000}\) (exactly the grid midpoint) carry only if the retained last bit is 1; otherwise don't. For rna carry whenever \(L\ge\texttt{0x8000}\) (ties go away from zero).
- A carry that ripples out of the mantissa into the exponent naturally pushes an oversized value to ±inf, so no extra overflow branch is needed.
- Special values: ±0 keeps its sign bit; ±inf is returned as is; NaN keeps its sign bit and forces the highest retained mantissa bit to 1 (it must still be NaN); a value already on the grid is returned unchanged by all three modes (idempotence).
- A non-float32 input or an unknown `rounding` raises ValueError.

## Example

On \([1,2)\) the bf16 grid step is \(2^{-7}=0.0078125\), and all four numbers below are **exactly** midpoints of neighboring grid points:

```
x = [1.00390625, 1.01171875, 1.01953125, 1.02734375]

rne -> [1.0, 1.015625, 1.015625, 1.03125]
rz  -> [1.0, 1.0078125, 1.015625, 1.0234375]
rna -> [1.0078125, 1.015625, 1.0234375, 1.03125]
```

rne alternates between the even neighbours across the four ties (1.0 and 1.015625 have retained last bit 0, while 1.0078125 and 1.0234375 have 1) — that sawtooth is precisely "to even"; rz always goes down and rna always goes up. The float32 maximum \(3.4028234663852886\times10^{38}\) overflows to `inf` under rne/rna and truncates to \(3.3895313892515355\times10^{38}\) under rz.

## Constraints

- float32 tensors of any shape, output is float32 of the same shape
- the oracle is the real hardware conversion `x.to(torch.bfloat16).to(torch.float32)` compared bitwise over normal, subnormal, \(10^{38}\) and \(10^{-38}\) magnitudes, with NaN positions compared separately
- general inequalities are also checked: \(|rz|\le|x|\), the absolute error of rne is no larger than that of rz, \(|rna|\ge|rz|\), and the relative error of rne stays below \(2^{-8}\) (half a grid step)
