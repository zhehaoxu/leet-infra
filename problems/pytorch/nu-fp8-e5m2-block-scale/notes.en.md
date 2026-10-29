# Notes: Why the Block Scale Is the Smallest Non-Overflowing Power of Two, and When the Subnormal Range Is Ever Reachable

## Origin

E5M2/E4M3 come from the OCP *FP8 Data Type Specification*, implemented by the H100 Transformer Engine and the MI300x FP8 path; OCP Microscaling (MX) encodes "one shared exponent per block" as an E8M0 block scale over blocks of 32, DeepSeek-V3 uses 128x128 and 1x128 tile-wise FP8 scaling for weights and activations, and block-quantized KV cache follows the same idea. The interview probes are: why the scale must be a power of two (multiplication adds no rounding), whether absmax scaling gets wrecked by a single outlier, and the E5M2 versus E4M3 trade-off.

## The Scale Definition Fixes Every Boundary

Specifying \(e\) as "the smallest integer with \(m\le57344\cdot2^e\)" buys two things:

- the absmax is always preserved exactly (never halved), and the other elements keep the best available resolution relative to the grid;
- the scale is a power of two, so `x / scale` and `q * scale` are **exact** and all quantization error comes from the single rounding step.

The classic way to break it is `e = ceil(log2(m / 57344))`: when \(m/\text{MAX}\) is exactly a power of two, the floating error in `log2` can return `k + 2e-16`, `ceil` gives \(k+1\), the scale doubles and the absmax is rounded down to the previous grid point. Do it with `math.frexp` instead: write \(m=f\cdot2^E\) with \(0.5\le f<1\), take \(e=E-1\) if \(f\) is exactly 0.5 and \(e=E\) otherwise; or simply loop-search from \(-127\) the way the oracle does.

## Ties-to-Even Must Be Decided on the Bit Pattern, Not the Value

E5M2 grid points are numbered contiguously by the pattern `E*4 + M` (subnormals occupy `0..3`, normals start at `E=1`), so "mantissa field \(M\) is even" is the same as "the pattern is even". The oracle enumerates every magnitude with its index, uses `bisect` to find the insertion point and compares the two neighbors: the strictly closer one wins, and on an exact tie the even-indexed side is taken. Testing "\(M\) is even" instead would require inverting the value back to \((E,M)\), handled separately for normals and subnormals — an easy place to go wrong.

## The Subnormal Range Is Unreachable When block=1

A counter-intuitive fact: with `block=1` the scale is perfectly adaptive (\(57344\cdot2^{e-1}<|x|\le57344\cdot2^e\)), so the normalized \(t\) always lands in \((28672,57344]\), which is nowhere near the \(2^{-16}\) subnormal lattice. To reach subnormals you need the block scale pinned by a normal-sized absmax while the block also holds a tiny element — e.g. `x = [1.0, 1.5*2^-31]` with `block=2`: absmax 1 gives \(e=-15\), scale \(2^{-15}\), the second element divided by the scale is \(1.5\cdot2^{-16}\), exactly halfway between the subnormal points \(1\cdot2^{-16}\) and \(2\cdot2^{-16}\), and ties-to-even takes \(M=2\), giving \(2^{-30}\). Writing that as its own case verifies the grid better than a hundred lines of random data.

## Core Ideas and Pitfalls

- **The expected value of `0.6`**: with `block=5` the nearest grid point to 0.6 is 0.625, not 0.5, because on the \(5\cdot2^{-13}\) grid it is closer to 0.625. Compute the scale first; do not guess from "E5M2 is very coarse".
- **Whole-block semantics for inf and NaN**: if the absmax is inf or the block contains NaN, the **entire block** outputs NaN. Deciding element-wise would turn `[inf, 1.0]` into `[inf, 1.0]` and clash with this repository's "non-finite values poison the block" convention (`nu-kahan-compensated-sum` degrades instead, because summation can fall back to a plain order, while quantization has nothing to skip to).
- **Do not lean on torch's saturation**: E5M2 overflows to inf, but the scale here guarantees it cannot, so the output contains neither inf nor saturated values. `x.to(torch.float8_e5m2)` agrees with the grid snap bitwise across normal and subnormal ranges, so it is a fine second oracle — it just will not pick your scale.
- **Clamping \(e\)**: the \([-127,127]\) range is insurance for extreme absmax values; after clamping, the absmax may exceed \(57344\cdot2^e\) and the quantized result lands on the top grid point rather than inf.
- **Sign of zero**: all output zeros are written as +0.0 so that `-0.0` cannot create noise in a bitwise comparison.

## How It Differs from Neighboring Problems

`fp8-quantize` handles E4M3 (3 mantissa bits, maximum 448, an explicit saturation rule) and is usually a single symmetric scale, while this problem is E5M2 plus per-block power-of-two scales; `quant-mx-microscaling` uses an E8M0 block scale that must be encoded into the output, and `quant-blockfp-tile` is about tile selection. The exclusive points here are the exact "smallest non-overflowing scale" test and ties on the subnormal lattice.

## Cross-Check Basis and Complexity

The oracle enumerates the grid, uses bisect with pattern parity for ties and a loop search for the scale — structurally different from the reference (frexp scaling + per-binade step + `torch.round`) — and requires bitwise equality. Coverage includes \(10^{-45}\), \(10^{38}\), random normals, exact \(57344\cdot2^k\) boundaries and `block` in 1/3/7/32/200. Time \(O(N)\) with one absmax reduction per block.
