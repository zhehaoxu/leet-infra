# FP8 (E4M3) Quantization Round Trip

## Problem

Since H100/MI300, FP8 has become the new default precision for training and inference. The most commonly used variant is **E4M3**: 1 sign bit, 4 exponent bits (bias 7), 3 mantissa bits. This problem **quantizes a floating-point tensor to the nearest representable E4M3 value and converts it back to floating point** (fake-quant round trip), requiring element-wise reproduction of IEEE-style "round to nearest, ties to even (RTNE)", and **saturation** to the maximum finite value when the value is out of range.

## E4M3 Format (follow this exactly)

With sign $s$, exponent field $E\in[0,15]$, and mantissa field $M\in[0,7]$:

- **Normal values** ($E\ge1$): $v=(-1)^s\cdot 2^{E-7}\cdot\left(1+\dfrac{M}{8}\right)$.
- **Subnormal values** ($E=0$): $v=(-1)^s\cdot 2^{-6}\cdot\dfrac{M}{8}=(-1)^s\cdot M\cdot 2^{-9}$ (a finer, equally spaced grid than the smallest normal number $2^{-6}=0.015625$, with spacing $2^{-9}$).
- **Maximum finite value $=448$** ($E=15, M=6$). $M=7$ with $E=15$ is not a number in this format (`float8_e4m3fn`, fn = no inf)—so any quantity whose rounding trend reaches $480$ **saturates** to $448$.

## Rules

Implement `solve(x)`, returning **the floating-point tensor after the quantization round trip** (same shape and dtype as `x`), element-wise:

1. $y=|x|$ (the sign is kept separately; `0` and NaN are returned as is).
2. If $2^{-6}\le y$ (normal region): the power-of-two binade $2^e\le y<2^{e+1}$ containing it has grid spacing $\text{step}=2^{e-3}$; $n=\operatorname{RTNE}(y/\text{step})$, $y'=n\cdot\text{step}$.
3. Otherwise (subnormal region $y<2^{-6}$): $\text{step}=2^{-9}$, $y'=\operatorname{RTNE}(y/\text{step})\cdot 2^{-9}$.
4. Saturation: if $y'>448$, use $448$.
5. Return $\operatorname{sign}(x)\cdot y'$.

$\operatorname{RTNE}$ = round half to **even** (consistent with `torch.round` / Python `round`).

## Examples

```
1.06   → 1.0          1.12  → 1.125        0.03  → 0.029296875
0.015  → 0.015625     2^-9  → 0.001953125  -1.06 → -1.0
449    → 448          1000  → 448 (saturated)    448   → 448
```

## Constraints

- The result must match `x.to(torch.float8_e4m3fn).to(x.dtype)` element-wise (built-in oracle)—do not use that cast; write the grid quantization by hand
- Input is finite floating point (float32/float64); output dtype and shape same as input; do not modify the input
- Common pitfalls: (1) when the exponent makes $y/step$ land in $[8,16)$ a carry occurs ($n=16$ → next binade); do not lock the binade; (2) subnormals use a **fixed** $2^{-9}$ spacing rather than per-binade spacing; (3) ties must round to even, not always up
