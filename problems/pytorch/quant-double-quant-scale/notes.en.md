# Quantization: Second-Level Quantization of Scales — Algorithm Notes

## 1. Why the scales need quantizing too

QLoRA stores weights as "4-bit codes + one FP32 absmax per 64 weights". The 4-bit part is 0.5 byte/weight, while the absmax part costs \(32/64=0.5\) bit per weight — **the scale overhead nearly equals half the code itself**. On a 65B model that is GB-scale on its own.

**Double quantization** (Dettmers et al., arXiv:2305.14314 §3) treats the first-level scale array as data and quantizes it again: every 256 FP32 scales share one FP32 second-level scale, and first-level scales drop to 8 bits. The overhead becomes

\[
\frac{8}{64}+\frac{32}{64\cdot256}=0.125+0.001953\approx0.127\ \text{bit/param}
\]

saving 0.373 bit/param over 0.5. This problem implements the **numerical core**: given first-level scales, encode and dequantize them at the second level.

## 2. Symmetric fixed point and grouping (judged point)

- **Symmetric**: first-level scales are always positive, so use signed fixed point over \([-qmax, qmax]\) with `qmax = 2^(k-1)-1` and zero-point 0 — no zero-point needed, dequantization is one multiplication;
- **Grouping**: `block_size` (QLoRA uses 256, matching `quantize_blockwise(..., blocksize=256)` in `compress_statistics`) **consecutive** scales form a group; a short trailing group stands alone;
- **Group max lands exactly on \(\pm qmax\)**: \(s^{(2)}=\max|s|/qmax\), so the largest-magnitude scale in every group is lossless — `check_error_bound` asserts it per group;
- **Error bound**: \(|\hat s-s|\le s^{(2)}/2\) (round's inherent error), verified with the \(s^{(2)}/2+10^{-7}\) bound in `check_error_bound`;
- **All-zero group**: \(s^{(2)}=1\), codes 0, reconstruction 0.

## 3. Division of labor with byte accounting

- `peft-qlora-nf4-forward`: computes only the **byte count** \(n/2+n/64+\lceil (n/64)/256\rceil\cdot4\), never the second-level numerics;
- This problem: only the **numerical round trip** (codes, second-level scale, reconstruction), no byte count; each problem is self-consistent.
- Note the convention difference: bitsandbytes actually stores first-level scales as **FP8** (e4m3); here the problem statement uniformly uses **symmetric fixed point** for exact element-wise judging — both are "8 bit/scale + one FP32 per 256" accounting, with different numerical paths.

## 4. Why group boundaries matter

`check_group_boundaries` uses extreme data — "first 256 entries 0.01, last 44 entries 1.0" — and asserts:

- the group count is 2 (the trailing 44 form their own group);
- group 0 has \(s^{(2)}=0.01/127\), group 1 has \(s^{(2)}=1/127\).

If grouping were implemented as "sort by value then bucket" or "merge the tail into the previous group", the two groups' \(s^{(2)}\) would contaminate each other (the 44 ones would inflate group 0's \(s^{(2)}\) by 100×) and the assertion fails immediately.

## 5. Implementation notes

- A per-group loop suffices (judging sizes ≤ 300); ragged vectorized grouping is unnecessary;
- Internal float64: float32 rounding of `s/s2` makes the "group max lossless" assertion jitter around the \(10^{-6}\) tolerance;
- `bits=8` gives `qmax=127`; `bits=2` gives `qmax=1` (the example uses it as the minimal anchor).

## 6. Verification

- `check_example`: four 0.25's, 2 bits, group width 2 → all codes 1, exact.
- `check_group_boundaries`: B=300 with group width 256 gives two independent groups.
- `check_error_bound`: error ≤ s2/2, group max lossless.
- `check_matches_oracle`: B=37, 4 bits, group width 8 (tail of 5), per-group recomputation.
- `check_zero_group_and_validation`: all-zero group + four kinds of invalid arguments.
