# Official mechanism and pitfalls

## Official source and version

- Repository: `thu-ml/SageAttention`, default branch `main` (the GitHub trees API returns 159 blobs, truncated=False).
- Files used: `csrc/fused/fused.cu` (`QuantInt8Kernel` / `MeanScaleKernel` / `TransposePadPermuteKernel` / `quant_per_block_int8_cuda` / `mean_scale_fuse_quant_cuda` / `scale_fuse_quant_cuda` / `transpose_pad_permute_cuda`), `csrc/numeric_conversion.cuh` (`float_to_int8_rn` / `floatx4_to_e4m3x4`), `sageattention/triton/quant_per_block.py` (`quant_per_block_int8_kernel`), `sageattention/quant.py` (`per_block_int8` / `per_channel_fp8`).
- How it was fetched: the GitHub API `git/trees/main?recursive=1` plus `contents?ref=main` plus `base64.b64decode`.
- The official cross-check script uses `.venv-eval/bin/python` (torch 2.14.1) to **expand both kernels line by line into equivalent pure Python**: one oracle each for the CUDA and the Triton INT8 formulas, one expansion each for `MeanScaleKernel` and `TransposePadPermuteKernel`, totalling **1800 groups** (int8 600 / fp8 600 / permute 600), all of which agree.

## The two per-block INT8 conventions

The repository ships **two** per-block INT8 implementations. They are mathematically equivalent but not point-wise equivalent:

| | CUDA `QuantInt8Kernel` | Triton `quant_per_block_int8_kernel` |
|---|---|---|
| `amax` | `max(0.0000001f, max\|x\|)` — **has a zero guard** | `max(\|x\|)` — **no lower bound** |
| Rounding | `cvt.rni.sat.s8.f32` (**round-to-nearest-even** + saturate) | `x_int8 += 0.5 * where(x_int8 >= 0, 1, -1)` (**round-half-away-from-zero**) |
| How the scaling is written | `x * (127/amax)` (`tmp_scale = 127.0f / s_amax`) | `x / (amax/127)` (`scale = tl.max(tl.abs(x)) / 127.`) |
| Out-of-range tokens | the out-of-range slot is set to `x_val_float = 0.0f` and takes part in the amax reduction | `tl.load` masks it off, so it does not take part in the max reduction |

- **Three places where RNE and half-away-from-zero part ways**:
  - positive `.5`: `2.5` gives `2` under RNE (the even one) but `3` under away-from-zero.
  - negative `.5`: `-2.5` gives `-2` under RNE (`floor(-2.5) = -3` is odd, so it steps up to `-2`) but `-3` under away-from-zero.
  - `-0.4`: the away convention subtracts `0.5` first, yielding `-0.9`, which **truncates toward zero** to `0` (not `-1`). Under RNE, `floor(-0.4) = -1` and `diff = 0.6 > 0.5` also give `0`.
- **The `sign` test uses the floating point value from before the `0.5` shift.** That is the semantics of `tl.where(x_int8 >= 0, 1, -1)` in the Triton source.
- **`sm_scale` is multiplied into the values first** (`x_val_float[i][j] *= sm_scale`); it feeds both the `amax` reduction and the quantization — it is not applied to the resulting integers.
- **The `1.44269504` factor on `sm_scale`**: the official `per_block_int8` passes `sm_scale * 1.44269504` (`log2(e)`) for the q side and `1.0` for the k side, because the Triton attention kernel implements softmax with exp2. That factor belongs to the caller; here `sm_scale` is an ordinary scaling factor.
- **Block boundaries**: the `q_scale` shape is `(qo_len + BLKQ - 1) // BLKQ` (`BLKQ = 128`) and the `k_scale` shape is `(kv_len + BLKK - 1) // BLKK` (`BLKK = 64`); in a partial last block only the real tokens take part in the quantization.
- **Saturation is unreachable in this problem's numeric range**: `|x| <= amax` and `scale = amax/127` imply `|x_int8| <= 127`, so the saturate branch of `cvt.rni.sat.s8.f32` never fires. Do not spend mutation budget there.

## Pitfalls of per-channel FP8 (E4M3)

- **`MeanScaleKernel` is per-channel.** `dim3 grid(num_heads, batch_size, head_dim)`, `d_id = blockIdx.z`: every `head_dim` channel reduces **independently** along the token dimension. This is the sharpest contrast with per-tensor quantization and the main judging point of this problem.
- **`fp8_padded_num_tokens = (num_tokens + 15) / 16 * 16`**: the mean's denominator is the length **after padding to 16** — not the real token count, and not the length after padding to 64 (that one is the shape of the output tensor).
- **The padding positions hold `0`**, filled by the upstream `TransposePadPermuteKernel` through `cp_async`'s `kFillZero`, and they **take part in the `max` / `min` / `sum` reduction**. Therefore the `min_val` of an all-positive channel is pulled down to `0`, which in turn changes `amax`.
- **With `smooth_v=True`, `amax` is the value after subtracting the mean**: `s_amax_val = fmaxf(fabsf(block_max_val - s_mean_val), fabsf(block_min_val - s_mean_val))`. Writing `max(|max|, |min|)` is wrong.
- **With `smooth_v=False` no mean is returned** (the official `scale_fuse_quant_cuda` passes `nullptr` for the mean, and the third element of the returned tuple is `None`), and `amax` is `max(|max|, |min|)`.
- **`scale` and `recp_scale` are reciprocal**: `scale = s_amax_val / scale_max` (`scale_max = 448.0`, the E4M3 upper bound) and `recp_scale = scale_max / s_amax_val`. The quantization multiplies by `recp_scale` (it does not divide by `scale`).
- **The FP8 bytes themselves are out of the judging scope**: `floatx4_to_e4m3x4` uses `cvt.rn.satfinite.e4m3x2.f32` (hardware rounding plus saturation to a finite value), which cannot be reproduced exactly on a CPU. Only `mean` / `amax` / `scale` / `recp_scale` are judged.
- The official `per_channel_fp8` returns `(v_fp8, v_scale, vm)`, where `vm` is `None` when `smooth_v=False`.

## Pitfalls of transpose-pad-permute

- **`padded_num_tokens = (num_tokens + pad_size - 1) / pad_size * pad_size`**, where `pad_size` is `CTA_SIZE = 64`.
- **Output layout**: with `tensor_layout="HND"` the shape is `(b, h_kv, head_dim, padded_num_tokens)`, i.e. the token dimension moves to the last.
- **The permutation formula**: `smem_load_row = base + (mod / 8) * 2 + ((mod / 2) % 4) * 4 + (mod % 2)`, where `base = ((thread_id / num_threads_per_token) / 16) * 16` is a **multiple of 16** and `mod = (thread_id / num_threads_per_token) % 16`. The permutation is therefore an **in-group permutation over groups of 16**, not over groups of 64.
- **Expanding the formula for `mod = 0..15` gives `0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15`**, which matches the kernel comment `// 0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15` **exactly**. The official `per_channel_fp8` docstring instead lists `0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15`, which is the **inverse permutation** (`out[j] = src[L1[j]]`) — both describe the same layout, just in opposite directions. This problem follows the kernel formula (source → destination).
- **The padding positions hold 0** (`cp_async::SharedMemFillMode::kFillZero`) and take part in the subsequent FP8 statistics.

## Verification approach and problem design

- Official cross-check: **600 int8 groups + 600 fp8 groups + 600 permute groups = 1800 groups, all agreeing**. The int8 part alternates between the CUDA and the Triton conventions, and the grid deliberately constructs `.5` boundaries (`-1.0, -0.5, 0.0, 0.5, 1.0, 2.5, -2.5`) to expose the difference between the two roundings.
- Test cases: 3 `check_*` functions, each with an independent pure Python oracle, hand-computed anchors, and a grid cross-check.
- The hand-computed anchors pin down: RNE taking the even integer at `.5` (`2.5 -> 2`), away-from-zero incrementing at `.5` (`2.5 -> 3`), the negative `.5` case (`-2.5` giving `-2` and `-3` respectively), `-0.4` truncating toward zero to `0`, the `amax_floor` giving `1e-7` versus `0` on an all-zero input, a partial last block computing its own `amax`, `sm_scale` influencing both `amax` and the quantization, two channels reducing independently in the per-channel mode, the mean's denominator being the length after padding to 16, the padding `0` taking part in the `min` reduction, `mean` being `None` when `smooth_v=False`, the real minimum when 16 all-positive tokens leave no padding, the permutation index sequence, the direction of the 16-element permutation, the difference between padding to 64 and to 16, and the fact that the transposed rows are the head_dim axis.
- Mutation testing: 49 single-point mutants, all CAUGHT, with a **preceding self-check that the correct solution is ACCEPTED in both modes**.

## Pitfalls hit during cross-checking and mutation (reusable lessons)

1. **The cross-check comparator must handle `inf`.** `recp_scale` is `inf` when `amax == 0`, and `abs(inf - inf) = nan` makes the assertion misfire. Check `math.isinf` first.
2. **The expected values in the cross-check script must match the solution's field set exactly.** The first version of the solution exposed a `saturated` field, but `|x| <= amax` always holds, so the quantized value never overflows and `saturated` is **always False** — a dead field. Exposing `scaled` (the floating point value that entered the rounder) is what gives it discriminating power: it makes the "multiply by `127/amax`" and "divide by `amax/127`" spellings visible.
3. **Going back to the official source can overturn your own design.** The first version of `fp8_channel_scale` performed a **global** reduction; only after checking `dim3 grid(num_heads, batch_size, head_dim)` in `MeanScaleKernel` did it become clear that it is **per-channel**, and the semantics of the whole mode had to be rewritten. **Read the kernel before writing the solution** — do not guess from the API name.
4. **The hand-written oracle must be independent of the solution's spelling.** `mean = None` and `means.append(None)` are two separate pieces of code in the oracle; mutating only one of them makes the mutant a no-op.
5. **True-equivalent mutants must be removed and replaced by measurable forms.** The first round had 7 survivors out of 48, all diagnosed as:
   - `rne-sat-lower` (the lower saturation bound): `|x_int8| <= 127` always holds, so the branch is unreachable;
   - `away-sign-flip` (`v >= 0` changed to `v > 0`): it differs only at `v == 0.0`, and `0.0` truncates to `0` on both sides;
   - `int8-cuda-mul-form` / `int8-triton-div-form`: `f * (127/amax)` and `f / (amax/127)` are nearly equal in floating point — the window is too narrow;
   - `fp8-nosmooth-amax`: `abs(v - 0.0) == abs(v)`;
   - `fp8-nosmooth-mean-not-none`: the anchor touched the dead variable `mean`, while the output used `means.append(None)`;
   - `fp8-min-init`: `max_val = vals[0]` is overwritten by the following loop with the same value.
   After replacing each with a measurable form, 49/49 were CAUGHT.
6. **The dimensions of a grid must be orthogonal.** The fp8 grid originally used `style = case % 4` with `smooth = case % 2`, which are permanently in phase (a `style` of 2 or 3 forces `smooth=True`), so half of the non-smooth-branch mutants went unmeasured. Switching to `style = case % 5` with `smooth = (case // 5) % 2` fixed the coverage.
7. **Mutants about initial values need purpose-built samples.** `max_val = 0.0` / `min_val = 0.0` are only measurable with an all-negative channel or with 16 all-positive tokens that leave no padding; two hand-computed anchors were added to catch them.

## Boundaries that stay out of the judging scope

The attention forward and online softmax, the accumulation order, the CUDA kernel's tile and thread mapping plus `cp_async` / shared memory / `ldmatrix` optimizations, the hardware rounding of the FP8 bytes (`cvt.rn.satfinite.e4m3x2.f32`), Triton compilation and launch configuration, the NHD branch of `tensor_layout`, the `per_warp_int8` warp variant, the `SubMeanKernel` and `km` (key mean) paths, and real weight matrices are all out of the judging scope. This problem keeps only the three deterministic pure numerical ledgers.
