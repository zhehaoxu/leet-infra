# Notes: Expressing RNE as One Addend, and Why NaN Rounds into inf

## Origin

Intel and Google pushed bf16 into training with the 2019 *GPU Assisted bfloat16 Matrices* paper; its winning property is sharing the same 8-bit exponent field as float32, so it needs no loss scaling and the conversion is just dropping mantissa bits. The bf16 units of T4/A100/H100, bfloat16 accumulation on TPUs, and the CPU simulation of "round to bf16 first, then compute" all rest on this rounding rule. The interview asks: bf16 versus fp16 trade-offs, why bf16 overflows to inf instead of saturating, and what statistical problem ties-to-even actually solves.

## One Addend Encodes Both "Nearest" and "Even"

Let \(L\) be the discarded low 16 bits and \(b=(u\gg16)\,\&\,1\) the retained last mantissa bit. Look at when \(u+0x7FFF+b\) carries into the high 16 bits:

- \(L>\texttt{0x8000}\): \(L+0x7FFF\ge\texttt{0x10000}\), always carries — the upper half of "nearest".
- \(L<\texttt{0x8000}\): \(L+0x7FFF\le\texttt{0xFFFE}\), never carries — the lower half of "nearest".
- \(L=\texttt{0x8000}\) (a tie): the sum is \(\texttt{0xFFFF}+b\), so it carries only when \(b=1\). And "carry when the last bit is 1" is exactly "the result becomes even after carrying", so ties-to-even needs no extra branch.

`rna` adds only 0x8000, so ties always carry (away from zero); `rz` adds nothing and is a pure mask. All three share the same final `& 0xFFFF0000`, which is why they match the hardware bit for bit.

## Core Ideas and Pitfalls

- **Treat the bit pattern as unsigned**: `x.view(torch.int32)` yields two's-complement values, a constant like `0xFFFF0000` is out of int32 range and triggers a promotion or an outright error in torch, and the test \(L\ge\texttt{0x8000}\) turns into "is \(L\) negative" under two's complement. The reference solution first does `.to(torch.int64) & 0xFFFFFFFF`, computes as if the 32 bits were unsigned, and returns to `int32`. Related trap: the pattern `0xFF800001` written as an int32 literal is **-8388607**, not -8394751.
- **NaN rounds into ±inf**: the pattern `0x7F800001` (a signaling NaN) plus 0x8000 becomes `0x7F808001`, whose high 16 bits are `0x7F80` — that is +inf. So NaN needs its own side path: `(u & 0xFFFF0000) | 0x00400000`, keeping the sign bit and forcing the quiet bit so the output stays NaN.
- **Do not write `round(x * 2**k) / 2**k`**: the bf16 grid step is \(2^{e-7}\) and depends on the exponent; a fixed scale only works inside one binade and breaks across \(2^e\) boundaries and in the subnormal range (zero exponent, fixed step \(2^{-133}\)). The bitwise method covers those regions for free, which is exactly why \(10^{-38}\), \(10^{38}\) and subnormals are all in the cross-check pool.
- **`torch.round` is already ties-to-even** (IEEE `rint` semantics), so "divide, round, multiply back" is fine as long as the step is right; the real difficulty is that the step has to be taken element-wise from the exponent, not the rounding rule itself.
- **\(10^{30}\) is not on the grid**: intuition says a power of ten must be exact, but bf16 keeps only 7 explicit mantissa bits and \(10^{30}\) genuinely moves. The test asserts `solve(g,'rne') != g` so that nobody passes by returning input unchanged.
- **Idempotence**: the output must still lie on the grid, so feeding it back through any mode has to be bitwise stable. This catches implementations that carry correctly but forget the mask.

## How It Differs from Neighboring Problems

`nu-stochastic-rounding` makes a probabilistic choice on **the same grid** (unbiased SR plus the fwd/bwd biases) while this problem is deterministic; `fp8-quantize` handles E4M3 with a different field split and an explicit saturation rule (capped at 448); `fp16-loss-scaling` is about a dynamic scale and underflow counters; `quant-mx-microscaling` shares one exponent per block and quantizes only the mantissa inside it. The exclusive point here is that the bitwise implementation must equal the hardware conversion bit for bit, including the NaN, subnormal and carry-overflow corners.

## Cross-Check Basis and Complexity

The oracle is the hardware path `x.to(torch.bfloat16).to(torch.float32)`: a completely independent implementation from the reference bitwise arithmetic, compared element by element (NaN positions aligned first, then `torch.equal` on the rest). The pool covers \(10^{38}\), \(10^{-38}\), subnormals, multiples of \(10^{30}\), and 5000 random values spanning 20 orders of magnitude. Complexity \(O(N)\), pure integer bit ops, no floating-point division.
