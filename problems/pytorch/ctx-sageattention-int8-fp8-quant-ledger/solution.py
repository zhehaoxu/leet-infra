"""order 840 参考解（暂存）：SageAttention 的 per-block INT8 / per-channel FP8 量化与转置置换账。

官方回源：thu-ml/SageAttention (main)
  csrc/fused/fused.cu::QuantInt8Kernel / MeanScaleKernel / TransposePadPermuteKernel
  csrc/numeric_conversion.cuh::float_to_int8_rn
  sageattention/triton/quant_per_block.py::quant_per_block_int8_kernel
  sageattention/quant.py::per_block_int8 / per_channel_fp8

三段纯数值账，不做 CUDA/Triton kernel、不做 attention 前向、不依赖 torch/numpy。
"""

import math

AMAX_FLOOR_CUDA = 0.0000001
FP8_SCALE_MAX = 448.0
LOG2E = 1.44269504


def _round_half_away_from_zero(v):
    """Triton 版：x += 0.5 * sign(x)，再向零截断到 int8。

    sign 判定用的是**加 0.5 之前**的浮点值；截断是向零（-0.9 -> 0）。
    """
    shifted = v + (0.5 if v >= 0 else -0.5)
    return int(shifted)  # Python int() 是向零截断


def _round_to_nearest_even(v):
    """CUDA 版：cvt.rni.sat.s8.f32 —— round-to-nearest-even 后 saturate 到 int8。"""
    lo = math.floor(v)
    diff = v - lo
    if diff > 0.5:
        r = lo + 1
    elif diff < 0.5:
        r = lo
    else:
        r = lo if lo % 2 == 0 else lo + 1
    if r > 127:
        r = 127
    elif r < -128:
        r = -128
    return r


def _int8_block_quant(x, blk, sm_scale, rounding, amax_floor):
    """per-block INT8 量化。

    x           : token × head_dim 的二维列表
    blk         : 每块 token 数
    sm_scale    : 缩放系数（官方 q 侧会再乘 log2(e)，此处由调用方传入）
    rounding    : "rne"（CUDA，round-to-nearest-even）或 "away"（Triton，half-away-from-zero）
    amax_floor  : amax 的防零下界（CUDA 为 1e-7，Triton 无下界取 0）
    """
    tokens = len(x)
    head_dim = len(x[0]) if tokens else 0
    out = []
    scaled = []
    scales = []
    amaxes = []

    num_blocks = (tokens + blk - 1) // blk
    for b in range(num_blocks):
        lo = b * blk
        hi = min(lo + blk, tokens)
        amax = amax_floor
        for t in range(lo, hi):
            for v in x[t]:
                av = abs(v * sm_scale)
                if av > amax:
                    amax = av
        scale = amax / 127.0
        scales.append(scale)
        amaxes.append(amax)
        # CUDA 版写 x * (127/amax)；Triton 版写 x / (amax/127)
        for t in range(lo, hi):
            for v in x[t]:
                f = v * sm_scale
                if rounding == "rne":
                    s = f * (127.0 / amax)
                    q = _round_to_nearest_even(s)
                else:
                    if scale == 0.0:
                        s = 0.0
                        q = 0
                    else:
                        s = f / scale
                        q = _round_half_away_from_zero(s)
                scaled.append(s)
                out.append(q)

    return {
        "amax": amaxes,
        "scale": scales,
        "scaled": scaled,
        "x_int8": out,
        "num_blocks": num_blocks,
        "head_dim": head_dim,
    }


def _fp8_channel_scale(x, scale_max, smooth_v):
    """per-channel FP8（E4M3）量化的 scale/mean 推导。

    官方 `MeanScaleKernel` 的 grid 是 (num_heads, batch_size, head_dim)，
    沿 **token 维**对**每个 head_dim 通道独立**做归约。

    归约区间长度是 padding 到 16 的 `fp8_padded_num_tokens`；padding 位置的值是 0
    （由上游 transpose_pad_permute 的 pad_zero 填充），会参与 max/min 归约，
    因此 min 通常会被拉到 0 或以下。

    均值分母是 padding 后的长度，不是真实 token 数。
    """
    tokens = len(x)
    head_dim = len(x[0]) if tokens else 0
    padded_tokens = ((tokens + 15) // 16) * 16
    pad_count = padded_tokens - tokens

    means = []
    amaxes = []
    scales = []
    recp_scales = []
    for d in range(head_dim):
        vals = [x[t][d] for t in range(tokens)]
        # padding 位置补 0，同样参与归约
        for _ in range(pad_count):
            vals.append(0.0)
        max_val = vals[0]
        min_val = vals[0]
        sum_val = 0.0
        for v in vals:
            if v > max_val:
                max_val = v
            if v < min_val:
                min_val = v
            sum_val += v
        if smooth_v:
            mean = sum_val / padded_tokens if padded_tokens else 0.0
            amax = max(abs(max_val - mean), abs(min_val - mean))
            means.append(mean)
        else:
            mean = None
            amax = max(abs(max_val), abs(min_val))
            means.append(None)
        scale = amax / scale_max
        recp_scale = scale_max / amax if amax != 0.0 else float("inf")
        amaxes.append(amax)
        scales.append(scale)
        recp_scales.append(recp_scale)

    return {
        "mean": means,
        "amax": amaxes,
        "scale": scales,
        "recp_scale": recp_scales,
        "padded_tokens": padded_tokens,
        "head_dim": head_dim,
    }


def _perm16():
    """官方 kernel 的置换算式（16 个一组）。

    源（组内位置 mod）-> 目标（组内位置 dest）：
    dest = base + (mod // 8) * 2 + ((mod // 2) % 4) * 4 + (mod % 2)

    逐项展开 mod = 0..15 得 dest 序列
        0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15
    与官方 kernel 注释一致；官方 `per_channel_fp8` docstring 给出的
        0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15
    是它的**逆置换**（out[j] = src[L1[j]]），描述的是同一布局。
    """
    return [(mod // 8) * 2 + ((mod // 2) % 4) * 4 + (mod % 2)
            for mod in range(16)]


def _transpose_pad_permute(x, pad_size):
    """转置 + token 维补齐 + fp8 MMA 布局的置换。

    输出布局：head_dim × padded_tokens。
    置换按 16 个一组：组内位置 mod 的 token 被搬到组内位置 perm16[mod]。
    """
    tokens = len(x)
    head_dim = len(x[0]) if tokens else 0
    padded_tokens = ((tokens + pad_size - 1) // pad_size) * pad_size

    # padded 矩阵：head_dim 行 × padded_tokens 列，padding 位置填 0
    padded = [[0.0] * padded_tokens for _ in range(head_dim)]
    for t in range(tokens):
        row = x[t]
        for d in range(head_dim):
            padded[d][t] = row[d]

    perm16 = _perm16()
    permuted = [[0.0] * padded_tokens for _ in range(head_dim)]
    for g in range(padded_tokens // 16):
        base = g * 16
        for mod in range(16):
            src = base + mod
            dst = base + perm16[mod]
            for d in range(head_dim):
                permuted[d][dst] = padded[d][src]

    return {
        "padded_tokens": padded_tokens,
        "padded": padded,
        "permuted": permuted,
        "permute_index": perm16,
    }


def solve(mode: str, **kw):
    if mode == "int8_block_quant":
        x = [list(r) for r in kw["x"]]
        return _int8_block_quant(
            x,
            int(kw.get("blk", 128)),
            float(kw.get("sm_scale", 1.0)),
            str(kw.get("rounding", "rne")),
            float(kw.get("amax_floor", AMAX_FLOOR_CUDA)),
        )
    if mode == "fp8_channel_scale":
        x = [list(r) for r in kw["x"]]
        return _fp8_channel_scale(
            x,
            float(kw.get("scale_max", FP8_SCALE_MAX)),
            bool(kw.get("smooth_v", True)),
        )
    if mode == "transpose_pad_permute":
        x = [list(r) for r in kw["x"]]
        return _transpose_pad_permute(x, int(kw.get("pad_size", 64)))
    raise ValueError("unknown mode: %s" % mode)
