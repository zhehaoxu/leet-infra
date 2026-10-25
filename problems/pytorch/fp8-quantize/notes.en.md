# FP8 E4M3 Quantization Round Trip Explained

## Why FP8, Why E4M3

FP8 stores floating point in one byte, halving memory/bandwidth and doubling tensor-core throughput; it is the approach used by H100 `transformer-engine` and DeepSeek training. There are two sub-formats: **E5M2** (wider range, lower precision, mostly used for gradients) and **E4M3** (one notch more precision, narrower range, mostly used for forward activations/weights). E4M3 has only 3 mantissa bits → relative precision of about $2^{-4}=6\%$; total error is kept in check by FP8 forward plus high-precision accumulation (FP32 accumulator).

## The Three Hard Parts of Hand-Written Grid Quantization

1. **Variable step per binade**: for normal values in $[2^e, 2^{e+1})$, the grid spacing is $2^{e-3}$ (3 mantissa bits → 8 grid points per binade). `frexp` is the most robust way to get the actual exponent; do not use `log2` then floor (boundary floating-point error picks the wrong binade).
2. **Fixed step for subnormals**: below $y<2^{-6}$ the spacing no longer shrinks with the binade; a uniform spacing of $2^{-9}$ (the standard result $=2^{E_{\min}-1-2}$) applies, guaranteeing equal spacing near 0 and a well-defined RTNE. Computing subnormals per binade is the most common bug.
3. **Saturation vs NaN**: the `fn` in `float8_e4m3fn` means "no inf"; $E{=}15,M{=}7$ is taken by NaN, and the maximum finite value is $=448$. RTNE would like to round to $480$ (which does not exist) → torch **saturates to 448**. Hence `465 → 448` rather than something larger.

## Round Half to Even (RTNE)

Python's built-in `round()` is exactly "round half to even", i.e. IEEE round-to-nearest-even: `round(0.5)=0`, `round(1.5)=2`, `round(2.5)=2`. Using it on `a/step` aligns with the torch cast. "Always round up" approaches such as `int(x+0.5)` differ from the oracle by one grid step at ties.

## Comparison with Integer Quantization

`symmetric/per-channel/group-wise` map a range onto **equally spaced integer grid points** (uniform quantization); FP8 uses a **non-uniform floating-point grid**—dense for small values, sparse for large values—naturally suited to data with a large dynamic range, and it needs no scale. `fp16-loss-scaling` deals with "overflow/underflow of low-bit-width floating point", which is the fp16 vs fp8 two sides of the same theme as the saturation/subnormals here.

## Judging

- Built-in oracle: `x.to(torch.float8_e4m3fn).to(dtype)`; `check_matches_oracle_*` cross-checks a large random grid + subnormals + saturation + ties (**exact equality**, both sides land on the same grid point value).
- `check_example` uses dumped concrete mappings; there are also float64, zero, and input-not-modified test cases.
