# Solution Notes · GEAR Low-Rank Power-Iteration Error Compensation

## In one sentence

Replace both ends of every row with the row mean, quantize that filled tensor, write the original extremes back, then run a power iteration whose QR steps happen only in the last round on the quantization residual: `output = quantized + low_rank`; the compensation is mathematically an orthogonal projection, so the error only shrinks.

## Why `low_rank` is a projection

Replacing the extremes by the row mean (rather than by 0 or by a trimmed mean) is only a placeholder that keeps the group's dynamic range from exploding; the original values are written back at the end, so those positions are represented exactly for free and their residual is identically 0. What actually enters the power iteration is the remaining error matrix \(A=X-\hat X\).

Both QR steps in the loop body are guarded by `if i == loop - 1`, so all earlier rounds are plain subspace iteration \(Q\leftarrow AP\), \(P\leftarrow A^{\top}Q\); when the loop ends the columns of \(Q\) are orthonormal and \(P=A^{\top}Q\), hence `low_rank` \(=QP^{\top}=QQ^{\top}A\) is the orthogonal projection of \(A\) onto \(\operatorname{span}(Q)\), \(A-\text{low\_rank}=(I-QQ^{\top})A\) is orthogonal to \(\operatorname{span}(Q)\), and \(\lVert A-\text{low\_rank}\rVert\le\lVert A\rVert\) always holds. For `loop=1` the subspace is \(\operatorname{span}(A\,\mathrm{qr}(P_0).Q)\); larger `loop` gets closer to the dominant right-singular subspace of \(A\) and the error decreases monotonically (example shape: 0.6166 → 0.3873 → 0.1925 → 0.1857 → 0.1856). This is also exactly why `q_base` is overwritten by `A @ P` in the first round and never affects the result.

## Three conventions that are easy to get wrong

1. **The budget is row-wide**: `n_out = int(int(B*H*L*D*sparsity)/B/L/2)` truncates twice and the divisor contains no `H`; `n_out` entries are taken at each end. Writing "n_out per head" or dropping the outer `int` is wrong.
2. **The mean includes the outliers**: the replacement value is the mean of the original row, not the trimmed mean after removing both ends; the whole row `[-100, 0, ..., 0, 100]` becomes 0.
3. **The level count inverted**: the denominator is `2**bits - 1` intervals (upstream `gears_tokenQ` typos it as `quantize_bit ** 2 - 1`, which only coincides for bits=2/4); a constant group does no division and dequantizes to `mn`.

## What is judged

- the mask: `n_out` candidates at each end of the merged row with ties going to the smaller index; when the two ends pick the same index the True count falls below \(2n_{\text{out}}\) (a constant row keeps only `n_out`);
- `quantized` carries the original values at outlier positions and `residual` is 0 there; every other point goes through the affine grid and rounding is half-to-even (in the statement's example, \((2-1)/(2/3)=1.5\) inside row0's head0 group lands on the even 2);
- `low_rank` is computed per \((b,h)\) block, has rank at most `rank`, and satisfies `||kv - output|| <= ||kv - quantized||`; swapping `q_base` or negating/rescaling `p_base` leaves it untouched, only a different `p_base` direction changes the subspace;
- all six keys present, dtype follows the input, a non-contiguous view matches a contiguous copy, and the three input tensors are never modified in place.

## Boundaries

- With `sparsity = 0` we get `n_out = 0`: the pipeline degenerates to "plain quantization + low-rank compensation" with an all-False mask (the example shape still drops its error from 2.108 to 0.745); a larger `sparsity` still caps out at half a row and never overflows.
- `group_size = H*D` merges the whole row into a single group; `bits = 1` leaves only the two grid points `mn`/`mx`, and the group extremes still round-trip exactly.
- The statement's example uses an all-ones `p_base` with `rank = 1`, so QR gives the direction \((1,\dots,1)/\sqrt{D}\); for a constant basis with `rank >= 2` the second QR column lies in the complement and its sign is implementation-defined, which is why the tests switch to full-rank random bases.

## Complexity

With merged row width \(C=H\cdot D\): two stable argsorts per row cost \(O(C\log C)\) and the quantization \(O(BLC)\); each power-iteration round costs two matmuls of \(O(LD\cdot\text{rank})\) per \((b,h)\) block, plus two QR steps in the last round, for \(O(\text{loop}\cdot B\,H\,L\,D\,\text{rank})\) in total.

## Source

`opengear-project/GEAR` (arXiv 2403.05527), local copy `/tmp/qsrc/GEAR/GenerationBench/GenerationTest/GEARLM/Simulated/compress_function.py`:

- `gears_tokenQ` L297-333: element count and budget L300-303 (`sparsity_pertoken = int(sparsity_num / batch / seq_len/2)`), merging into token rows L304-306, `torch.topk` at both ends L309/L311, row mean L312, the two replacing `scatter_` calls L316/L319, the quantizer call L321-322, writing originals back L327-328;
- `fake_groupwise_token_asymmetric_quantization_cluster` L132-160: group split L142, group extremes L144, `scale = (mx - mn) / cluster_num` L147, `(x - mn)/scale` L148, `F.relu` L149, `round_()` L150, dequantization L151;
- `fake_poweriteration_group` L69-98: `for i in range(loop)` L86, the two `torch.linalg.qr(...).Q` calls guarded by the last round L87-88 and L90-91, `q_base[0] = input @ p_base[0]` L89, `p_base[0] = transpose(input, 2, 3) @ q_base[0]` L92, `input = q_base[0] @ transpose(p_base[0], 2, 3)` L93;
- `gearslkivi_tokenQ_new` L204-211: L207 calls `gears_tokenQ`, L209 `error = cloned_input - output`, L210 the power iteration, L211 `return output + error_lr`;
- `compress_insert_function` L421-584: the GEAR branch L466-495 — K goes through the channel-wise `gearslkivi_channelQ_new` (L477) then `.half()` (L486), V through the token-wise `gearslkivi_tokenQ_new` (L487) then `.half()` (L495).

**Additions of this problem** (deviations from upstream; do not attribute them to the paper):

1. Upstream `gears_tokenQ` L322 passes the level count as `quantize_bit ** 2 - 1` (same typo in `gears_channelQ` L286); it agrees with the intended `2 ** bits - 1` (the non-cluster versions at L24 and L55 of the same file) only for `bits ∈ {2, 4}`. This problem always uses `2 ** bits - 1`.
2. Upstream generates the two bases with `torch.rand` inside the function (L83-84; L72-73 are the commented-out older form); here they are explicit `p_base`/`q_base` arguments, which makes the result deterministic and provably ignores `q_base` for `loop >= 1`.
3. Upstream divides by zero and returns NaN for a constant group (L147 has no guard); here the scale is 0, the code is always 0 and the value dequantizes to the group min.
4. Upstream relies on `torch.topk` (L309/L311), whose tie order is undefined; here ties go to the smaller index (stable argsort) as a determinism contract.
5. Upstream casts the result with `.half()` (L332); here the output dtype follows the input, with everything computed in float64 internally.
6. Only the token-wise (V-side) variant is exercised; the channel-wise K variant groups along a different axis (`gears_channelQ` L269-271 starts with `permute(0, 1, 3, 2)`) and is out of scope.

## Related problems

- `quant-kivi-kv-asym`: the quantizer of stage 2 here is exactly KIVI-style token-wise asymmetric group quantization, but KIVI stops there and never deals with the leftover error.
- `quant-qaq-error-bitwidth`: also picks outliers out, but QAQ lets them pass through with their original values and compensates nothing, while GEAR additionally uses a low-rank term to absorb part of the non-outlier error.
- `quant-rtn-gptq-error`: GPTQ absorbs error column by column with second-order information *while* quantizing; this problem compensates *after* quantizing with a rank-`rank` projection.
