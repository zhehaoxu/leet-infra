# SageAttention: the per-block INT8 and per-channel FP8 quantization, rounding, and transpose-permute ledger

## Background

SageAttention quantizes the two large matrices in an attention operator (query / key) into INT8 or FP8 before handing them to the Tensor Cores. The numerical details of that quantization step decide the final accuracy, and those details differ between the two implementations in the repository:

- The **CUDA kernel** (`csrc/fused/fused.cu`) rounds with the hardware instruction `cvt.rni.sat.s8.f32`, i.e. round-to-nearest-**even**.
- The **Triton kernel** (`sageattention/triton/quant_per_block.py`) uses `x += 0.5 * sign(x)` followed by truncation, i.e. round-half-**away-from-zero**.

The two disagree on `.5` boundaries, and they diverge further on negative values. The FP8 path is **per-channel**: it computes `mean` / `amax` along the token dimension independently for every `head_dim` channel, then derives `scale` and `recp_scale` from them. The FP8 mma layout additionally requires a "transpose + pad + 16-element tile permutation" pass.

This problem rebuilds those three **pure numerical** ledgers: the two rounding conventions of per-block INT8, the per-channel FP8 statistics, and the transpose-pad-permute layout. Attention forward, CUDA/Triton kernel scheduling, and the hardware rounding of the FP8 bytes are out of scope — every input is a plain list of numbers that has already been computed.

## General conventions

- Every input is a list of Python `float`s (`x` is a 2-D list: token × head_dim); no tensors, gradients, or batching.
- Floating point comparison uses absolute tolerance `1e-9` plus relative tolerance `1e-9`.
- `head_dim` is written `H` (number of channels); `tokens` is written `L`.
- Return values are always a `dict`; the field names are given per mode below.
- All indices are 0-based and **inclusive**.
- Every mode is reached through the single entry point `solve(mode, **kw)`.

## Modes and rules

### 1. `int8_block_quant`: per-block INT8 quantization

Inputs:

| Parameter | Meaning | Default |
|---|---|---|
| `x` | 2-D list of length `L`, each row of length `H` | required |
| `blk` | number of tokens per block | `128` |
| `sm_scale` | scaling factor (multiplied into the values) | `1.0` |
| `rounding` | `"rne"` (CUDA) or `"away"` (Triton) | `"rne"` |
| `amax_floor` | lower bound protecting `amax` against zero | `0.0000001` |

Steps:

```
# 1. multiply sm_scale into every value first
f[t][d] = x[t][d] * sm_scale

# 2. block along the token dimension; each block has its own amax and scale
num_blocks = (L + blk - 1) // blk
for b in range(num_blocks):
    lo = b * blk
    hi = min(lo + blk, L)                       # the last block may be partial
    amax[b] = max(amax_floor, max(|f[t][d]| for t in [lo,hi), d in [0,H)))
    scale[b] = amax[b] / 127

# 3. quantize point by point
rounding == "rne":                              # CUDA convention
    s = f * (127 / amax)
    q = round_to_nearest_even(s)                # then saturate to [-128, 127]
rounding == "away":                             # Triton convention
    s = f / scale                               # equivalent to f * (127 / amax)
    q = trunc(s + 0.5 * sign(s))                # sign is taken from s before the shift
```

Returned fields:

- `amax`: length `num_blocks`, the `amax` of each block;
- `scale`: length `num_blocks`, the `scale` of each block;
- `scaled`: length `L * H`, the **floating point value before the rounder** (flattened token-major);
- `x_int8`: length `L * H`, the quantized integers (flattened token-major);
- `num_blocks`: number of blocks;
- `head_dim`: number of channels.

Pitfalls:

- **The two rounding conventions differ on `.5` boundaries.** `round_to_nearest_even` picks the **even** integer at `x.5` (`2.5 -> 2`, `3.5 -> 4`); `trunc(s + 0.5*sign(s))` is half-away-from-zero (`2.5 -> 3`, `-2.5 -> -3`).
- **`sign(s)` is taken from `s` before adding `0.5`.** `-0.4` takes the negative branch, becomes `-0.9`, and truncates toward zero to `0` (not `-1`).
- **Triton truncates toward zero**: `trunc(-0.9) = 0`, `trunc(-1.0) = -1`. Python's `int()` happens to truncate toward zero and can be used directly.
- **CUDA has a `1e-7` lower bound on `amax`**, Triton does not. For an all-zero input CUDA's `amax` is `1e-7` while Triton's is `0` (then `scale` is `0` and the quantized value is recorded as `0`).
- **`sm_scale` is multiplied into the values first**; it affects both `amax` and the quantized result — it is not applied to the resulting integers.
- The official `per_block_int8` passes `sm_scale * 1.44269504` (`log2(e)`) for the q side and `1.0` for the k side; that factor is the caller's business, and here `sm_scale` is an ordinary scaling factor.
- **`scaled` is the field that exposes the intermediate quantity**: it makes the difference between the "multiply by `127/amax`" and "divide by `amax/127`" spellings visible, even though the two are mathematically equivalent.

### 2. `fp8_channel_scale`: per-channel FP8 (E4M3) statistics

Inputs:

| Parameter | Meaning | Default |
|---|---|---|
| `x` | 2-D list of length `L`, each row of length `H` | required |
| `scale_max` | maximum magnitude of FP8 (the E4M3 upper bound) | `448.0` |
| `smooth_v` | whether to subtract the mean | `True` |

Steps:

```
# 1. length of the reduction range: padded to a multiple of 16
padded_tokens = ((L + 15) // 16) * 16
pad_count = padded_tokens - L

# 2. every head_dim channel reduces independently (along the token dimension)
for d in range(H):
    vals = [x[t][d] for t in range(L)] + [0.0] * pad_count
    max_val = max(vals)
    min_val = min(vals)
    sum_val = sum(vals)
    if smooth_v:
        mean[d] = sum_val / padded_tokens
        amax[d] = max(|max_val - mean[d]|, |min_val - mean[d]|)
    else:
        mean[d] = None
        amax[d] = max(|max_val|, |min_val|)
    scale[d] = amax[d] / scale_max
    recp_scale[d] = scale_max / amax[d]
```

Returned fields:

- `mean`: length `H`; every element is `None` when `smooth_v=False`;
- `amax`: length `H`;
- `scale`: length `H`;
- `recp_scale`: length `H`; `inf` when `amax == 0`;
- `padded_tokens`: the padded length;
- `head_dim`: number of channels.

Pitfalls:

- **It is per-channel, not per-tensor**: every `head_dim` channel reduces independently along the token dimension. For `x = [[1.0, 10.0], [3.0, 20.0]]` the two channels get completely different `mean` and `amax`.
- **The mean's denominator is `padded_tokens` (the length after padding to 16), not the real token count.** For `L = 1` the denominator is `16`, so `mean` is `sum / 16`.
- **The padding positions hold `0` and take part in the `max` / `min` / `sum` reduction.** Therefore the `min_val` of an all-positive channel is pulled down to `0` whenever `L` is not a multiple of 16.
- **With `smooth_v=True`, `amax` is the value after subtracting the mean**: `max(|max_val - mean|, |min_val - mean|)`, not `max(|x|)`. This is the single easiest step to get wrong.
- **With `smooth_v=False`, `mean` is `None`** (the official code returns no mean tensor) and `amax` is `max(|max_val|, |min_val|)`.
- **`scale` and `recp_scale` are reciprocal**: `scale = amax / scale_max`, `recp_scale = scale_max / amax`. The quantization multiplies by `recp_scale`.
- `padded_tokens` pads to **16**, not to 64 (64 is the padding of the output tensor, see mode 3).

### 3. `transpose_pad_permute`: transpose + pad + FP8 mma layout permutation

Inputs:

| Parameter | Meaning | Default |
|---|---|---|
| `x` | 2-D list of length `L`, each row of length `H` | required |
| `pad_size` | granularity the token dimension is padded to | `64` |

Steps:

```
# 1. pad the token dimension to a multiple of pad_size; padding holds 0
padded_tokens = ((L + pad_size - 1) // pad_size) * pad_size

# 2. transpose into head_dim × padded_tokens
padded[d][t] = x[t][d]          # the original value for t < L, 0 otherwise

# 3. permute in groups of 16
#    source (in-group position mod) -> destination (in-group position dest):
#    dest = base + (mod // 8) * 2 + ((mod // 2) % 4) * 4 + (mod % 2)
#    where base is a multiple of 16
permuted[d][base + dest_off] = padded[d][base + mod]
```

Returned fields:

- `padded_tokens`: the padded length;
- `padded`: 2-D list of `H` rows × `padded_tokens` columns, the transposed result **before** the permutation;
- `permuted`: 2-D list of `H` rows × `padded_tokens` columns, the result **after** the permutation;
- `permute_index`: a list of 16 integers, the in-group permutation offsets.

`permute_index` expanded item by item (`mod = 0..15`) is:

```
0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15
```

Pitfalls:

- **The permutation is an in-group permutation over groups of 16**, not over groups of 64, and not a global permutation.
- **The direction is source → destination**: `permuted[d][base + permute_index[mod]] = padded[d][base + mod]`. Reading `permute_index` the other way round (as "destination → source") yields the inverse permutation, which is still a legal arrangement but is **not** the official one.
- **The padding granularity `pad_size` and the FP8 reduction's padding to 16 are two different things**: the former is the shape of the output tensor (default 64), the latter is the length of the statistics range (fixed 16).
- **Padding positions hold 0** and take part in the subsequent FP8 statistics.

## Judging scope

- Only the three pure numerical ledgers above are judged: `int8_block_quant` / `fp8_channel_scale` / `transpose_pad_permute`.
- **Not judged**: the attention forward, online softmax, accumulation, the CUDA kernel's tile and thread mapping, `cp_async` and shared memory, the `ldmatrix` optimization, the hardware rounding of the FP8 bytes (`cvt.rn.satfinite.e4m3x2.f32`), Triton compilation and launch configuration, the NHD branch of `tensor_layout`, the `per_warp_int8` warp variant, real weight matrices.
- The `x` in the problem statement is just a list of numbers and carries **no real weights**.
