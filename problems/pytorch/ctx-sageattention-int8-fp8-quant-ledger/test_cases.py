"""order 840 判题用例（暂存）：SageAttention 的 per-block INT8 / per-channel FP8 量化与转置置换账。

独立 oracle（不 import solution），每个 check_* 含手算锚点 + 网格对拍。
"""

import math


# ------------------------------------------------------------------ 独立 oracle
def oracle_rne(v):
    """CUDA `cvt.rni.sat.s8.f32`：round-to-nearest-even + saturate。"""
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


def oracle_away(v):
    """Triton：x += 0.5*sign(x)（sign 用加之前的值），再向零截断。"""
    shifted = v + (0.5 if v >= 0 else -0.5)
    if shifted >= 0:
        return int(shifted)
    return -int(-shifted)


def oracle_int8_block(x, blk, sm_scale, rounding, amax_floor):
    tokens = len(x)
    head_dim = len(x[0]) if tokens else 0
    num_blocks = (tokens + blk - 1) // blk
    amaxes = []
    scales = []
    scaled = []
    out = []
    for b in range(num_blocks):
        lo = b * blk
        hi = min(lo + blk, tokens)
        vals = []
        for t in range(lo, hi):
            for v in x[t]:
                vals.append(v * sm_scale)
        amax = amax_floor
        for v in vals:
            av = abs(v)
            if av > amax:
                amax = av
        scale = amax / 127.0
        amaxes.append(amax)
        scales.append(scale)
        for v in vals:
            if rounding == "rne":
                s = v * (127.0 / amax)
                q = oracle_rne(s)
            else:
                if scale == 0.0:
                    s = 0.0
                    q = 0
                else:
                    s = v / scale
                    q = oracle_away(s)
            scaled.append(s)
            out.append(q)
    return {"amax": amaxes, "scale": scales, "scaled": scaled,
            "x_int8": out, "num_blocks": num_blocks, "head_dim": head_dim}


def oracle_fp8_channel(x, scale_max, smooth_v):
    tokens = len(x)
    head_dim = len(x[0]) if tokens else 0
    padded_tokens = ((tokens + 15) // 16) * 16
    pad = padded_tokens - tokens
    means = []
    amaxes = []
    scales = []
    recp_scales = []
    for d in range(head_dim):
        vals = []
        for t in range(tokens):
            vals.append(x[t][d])
        for _ in range(pad):
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
    return {"mean": means, "amax": amaxes, "scale": scales,
            "recp_scale": recp_scales, "padded_tokens": padded_tokens,
            "head_dim": head_dim}


def oracle_perm16():
    return [(mod // 8) * 2 + ((mod // 2) % 4) * 4 + (mod % 2)
            for mod in range(16)]


def oracle_transpose(x, pad_size):
    tokens = len(x)
    head_dim = len(x[0]) if tokens else 0
    padded_tokens = ((tokens + pad_size - 1) // pad_size) * pad_size
    padded = [[0.0] * padded_tokens for _ in range(head_dim)]
    for t in range(tokens):
        for d in range(head_dim):
            padded[d][t] = x[t][d]
    perm16 = oracle_perm16()
    permuted = [[0.0] * padded_tokens for _ in range(head_dim)]
    for g in range(padded_tokens // 16):
        base = g * 16
        for mod in range(16):
            src = base + mod
            dst = base + perm16[mod]
            for d in range(head_dim):
                permuted[d][dst] = padded[d][src]
    return {"padded_tokens": padded_tokens, "padded": padded,
            "permuted": permuted, "permute_index": perm16}


# ------------------------------------------------------------------ 工具
def _close(actual, expect, atol=1e-9, rtol=1e-9, tag=""):
    if isinstance(expect, (list, tuple)):
        assert isinstance(actual, (list, tuple)), "%s: expected list got %r" % (tag, type(actual))
        assert len(actual) == len(expect), "%s: length %d != %d" % (tag, len(actual), len(expect))
        for i, item in enumerate(expect):
            _close(actual[i], item, atol=atol, rtol=rtol, tag="%s[%d]" % (tag, i))
        return
    if expect is None:
        assert actual is None, "%s: got %r expected None" % (tag, actual)
        return
    if isinstance(expect, bool) or isinstance(actual, bool):
        assert bool(actual) == bool(expect), "%s: got %r expected %r" % (tag, actual, expect)
        return
    if isinstance(expect, int):
        assert int(actual) == int(expect), "%s: got %r expected %r" % (tag, actual, expect)
        return
    a = float(actual)
    e = float(expect)
    if e != e:
        assert a != a, "%s: got %r expected NaN" % (tag, actual)
        return
    if a != a:
        raise AssertionError("%s: got NaN expected %r" % (tag, e))
    if math.isinf(e):
        assert math.isinf(a) and (a > 0) == (e > 0), \
            "%s: got %r expected %r" % (tag, actual, expect)
        return
    assert abs(a - e) <= atol + rtol * abs(e), \
        "%s: got %r expected %r" % (tag, actual, expect)


def _cmp(got, exp, tag):
    assert isinstance(got, dict), "%s: expected dict got %r" % (tag, type(got))
    assert set(got.keys()) == set(exp.keys()), "%s: keys %r != %r" % (
        tag, sorted(got.keys()), sorted(exp.keys()))
    for k in exp:
        _close(got[k], exp[k], tag=tag + "." + k)


class _Rng:
    """确定性 LCG，避免依赖 random/numpy。"""

    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def next(self):
        self.s = (1103515245 * self.s + 12345) & 0x7FFFFFFF
        return self.s / float(0x7FFFFFFF)

    def uniform(self, lo, hi):
        return lo + (hi - lo) * self.next()

    def randint(self, lo, hi):
        return lo + int(self.next() * (hi - lo + 1)) % (hi - lo + 1)

    def choice(self, seq):
        return seq[int(self.next() * len(seq)) % len(seq)]


# ------------------------------------------------------------------ check_*
def check_int8_block_quant(solve):
    # 手算锚点 1：CUDA 口径，RNE 的 .5 边界取偶数
    # x=[[1.0, 0.5]], blk=1 -> 每块 1 个 token，amax = 1.0
    # scale = 1/127 ; tmp_scale = 127/1 = 127
    # 1.0*127 = 127.0 -> 127 ; 0.5*127 = 63.5 -> RNE 取偶 -> 64
    r = solve("int8_block_quant", x=[[1.0, 0.5]], blk=1, sm_scale=1.0,
              rounding="rne", amax_floor=0.0000001)
    _cmp(r, {"amax": [1.0], "scale": [1.0 / 127.0],
             "scaled": [127.0, 63.5], "x_int8": [127, 64],
             "num_blocks": 1, "head_dim": 2}, "int8.anchor1")

    # 手算锚点 2：Triton 口径，half-away-from-zero 的 .5 边界进位
    # 同上 x，away：0.5*127=63.5 -> +0.5 = 64.0 -> 64（与 RNE 同）
    # 改用 x=[[1.0, 0.25]]：0.25*127 = 31.75 -> 32.25 -> 32
    # 用 2.5 造边界：x=[[1.0, 2.0/254.0]] -> (2/254)*127 = 1.0 -> away: 1.0+0.5=1.5 -> 1
    # 更直接：x=[[1.0, 0.5]] with blk=1 但 amax 取 1.0 -> 上面的
    # 用 x=[[2.0, 1.0]] blk=1：块 0 amax=2.0，scaled = 2*63.5=127.0, 1*63.5=63.5
    #   RNE: 63.5 -> 64 ; away: 63.5+0.5=64.0 -> 64   （仍相同）
    # 取 scaled 恰为 2.5：需要 x*127/amax = 2.5 -> x = amax*2.5/127
    #   amax=127.0 时 x=2.5 : scaled=2.5 -> RNE -> 2 ; away -> 3
    r = solve("int8_block_quant", x=[[127.0, 2.5]], blk=1, sm_scale=1.0,
              rounding="rne", amax_floor=0.0000001)
    _cmp(r, {"amax": [127.0], "scale": [1.0], "scaled": [127.0, 2.5],
             "x_int8": [127, 2], "num_blocks": 1, "head_dim": 2},
         "int8.anchor2a")
    r = solve("int8_block_quant", x=[[127.0, 2.5]], blk=1, sm_scale=1.0,
              rounding="away", amax_floor=0.0)
    _cmp(r, {"amax": [127.0], "scale": [1.0], "scaled": [127.0, 2.5],
             "x_int8": [127, 3], "num_blocks": 1, "head_dim": 2},
         "int8.anchor2b")

    # 手算锚点 3：负数的 .5 边界 —— RNE 与 away 在 -2.5 处不同
    # x=[[127.0, -2.5]] -> scaled = -2.5
    # RNE: floor(-2.5) = -3, diff = 0.5 -> 取偶 -> -2
    # away: -2.5 - 0.5 = -3.0 -> 向零截断 -> -3
    r = solve("int8_block_quant", x=[[127.0, -2.5]], blk=1, sm_scale=1.0,
              rounding="rne", amax_floor=0.0000001)
    _close(r["x_int8"], [127, -2], tag="int8.anchor3a.x_int8")
    r = solve("int8_block_quant", x=[[127.0, -2.5]], blk=1, sm_scale=1.0,
              rounding="away", amax_floor=0.0)
    _close(r["x_int8"], [127, -3], tag="int8.anchor3b.x_int8")

    # 手算锚点 4：-0.4 在 away 口径下向零截断为 0（不是 -1）
    # x=[[127.0, -0.4]] -> scaled = -0.4 ; away: -0.4-0.5 = -0.9 -> 0
    # RNE: floor(-0.4) = -1, diff = 0.6 > 0.5 -> 0
    r = solve("int8_block_quant", x=[[127.0, -0.4]], blk=1, sm_scale=1.0,
              rounding="away", amax_floor=0.0)
    _close(r["x_int8"], [127, 0], tag="int8.anchor4.x_int8")

    # 手算锚点 5：amax_floor 的作用 —— 全零输入时 CUDA 的 amax 是 1e-7 而非 0
    r = solve("int8_block_quant", x=[[0.0, 0.0]], blk=1, sm_scale=1.0,
              rounding="rne", amax_floor=0.0000001)
    _cmp(r, {"amax": [0.0000001], "scale": [0.0000001 / 127.0],
             "scaled": [0.0, 0.0], "x_int8": [0, 0],
             "num_blocks": 1, "head_dim": 2}, "int8.anchor5a")
    r = solve("int8_block_quant", x=[[0.0, 0.0]], blk=1, sm_scale=1.0,
              rounding="away", amax_floor=0.0)
    _cmp(r, {"amax": [0.0], "scale": [0.0], "scaled": [0.0, 0.0],
             "x_int8": [0, 0], "num_blocks": 1, "head_dim": 2},
         "int8.anchor5b")

    # 手算锚点 6：分块 —— blk=2 时 3 个 token 分成 2 块，每块各自算 amax
    # x=[[4.0],[1.0],[2.0]] blk=2 sm_scale=1
    # 块 0: token 0,1 -> amax = 4.0, scale = 4/127, scaled = 4*127/4=127, 1*127/4=31.75
    #   RNE: 127 -> 127 ; 31.75 -> 32
    # 块 1: token 2 -> amax = 2.0, scale = 2/127, scaled = 2*127/2 = 127 -> 127
    r = solve("int8_block_quant", x=[[4.0], [1.0], [2.0]], blk=2,
              sm_scale=1.0, rounding="rne", amax_floor=0.0000001)
    _cmp(r, {"amax": [4.0, 2.0], "scale": [4.0 / 127.0, 2.0 / 127.0],
             "scaled": [127.0, 31.75, 127.0], "x_int8": [127, 32, 127],
             "num_blocks": 2, "head_dim": 1}, "int8.anchor6")

    # 手算锚点 7：sm_scale 参与 amax 与量化（乘在值上，不是乘在结果上）
    # x=[[2.0],[4.0]] blk=2 sm_scale=0.5
    # 值 = [1.0, 2.0] -> amax = 2.0 -> scaled = 1.0*63.5 = 63.5 -> RNE 64
    #                                   2.0*63.5 = 127.0 -> 127
    r = solve("int8_block_quant", x=[[2.0], [4.0]], blk=2, sm_scale=0.5,
              rounding="rne", amax_floor=0.0000001)
    _cmp(r, {"amax": [2.0], "scale": [2.0 / 127.0],
             "scaled": [63.5, 127.0], "x_int8": [64, 127],
             "num_blocks": 1, "head_dim": 1}, "int8.anchor7")

    # 网格对拍
    rng = _Rng(84011)
    for case in range(150):
        style = case % 6
        tokens = rng.randint(1, 10)
        head_dim = rng.randint(1, 4)
        blk = rng.randint(1, 6)
        if style == 0:
            x = [[rng.uniform(-1.0, 1.0) for _ in range(head_dim)]
                 for _ in range(tokens)]
            sm_scale = 1.0
        elif style == 1:
            x = [[rng.uniform(-4.0, 4.0) for _ in range(head_dim)]
                 for _ in range(tokens)]
            sm_scale = 0.5
        elif style == 2:
            x = [[rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0, 2.5, -2.5])
                  for _ in range(head_dim)] for _ in range(tokens)]
            sm_scale = 1.0
        elif style == 3:
            x = [[rng.choice([-0.5, 0.5, 1.5, -1.5, 2.5, -2.5, 3.5])
                  for _ in range(head_dim)] for _ in range(tokens)]
            sm_scale = 1.0
        elif style == 4:
            x = [[0.0 for _ in range(head_dim)] for _ in range(tokens)]
            sm_scale = 1.0
        else:
            x = [[rng.uniform(-100.0, 100.0) for _ in range(head_dim)]
                 for _ in range(tokens)]
            sm_scale = rng.uniform(0.01, 2.0)
        if case % 2 == 0:
            got = solve("int8_block_quant", x=x, blk=blk, sm_scale=sm_scale,
                        rounding="rne", amax_floor=0.0000001)
            exp = oracle_int8_block(x, blk, sm_scale, "rne", 0.0000001)
        else:
            got = solve("int8_block_quant", x=x, blk=blk, sm_scale=sm_scale,
                        rounding="away", amax_floor=0.0)
            exp = oracle_int8_block(x, blk, sm_scale, "away", 0.0)
        _cmp(got, exp, "int8.grid%d" % case)
    print("int8_block_quant OK")


def check_fp8_channel_scale(solve):
    # 手算锚点 1：per-channel —— 两个通道各自独立归约
    # x=[[1.0, 10.0],[3.0, 20.0]] tokens=2 -> padded_tokens = 16
    # 通道 0: vals = [1.0, 3.0] + 14 个 0 -> max=3, min=0, sum=4
    #   mean = 4/16 = 0.25 ; amax = max(|3-0.25|, |0-0.25|) = 2.75
    #   scale = 2.75/448 ; recp_scale = 448/2.75
    # 通道 1: vals = [10.0, 20.0] + 14 个 0 -> max=20, min=0, sum=30
    #   mean = 30/16 = 1.875 ; amax = max(18.125, 1.875) = 18.125
    r = solve("fp8_channel_scale", x=[[1.0, 10.0], [3.0, 20.0]],
              scale_max=448.0, smooth_v=True)
    _cmp(r, {
        "mean": [0.25, 1.875],
        "amax": [2.75, 18.125],
        "scale": [2.75 / 448.0, 18.125 / 448.0],
        "recp_scale": [448.0 / 2.75, 448.0 / 18.125],
        "padded_tokens": 16,
        "head_dim": 2,
    }, "fp8.anchor1")

    # 手算锚点 2：smooth_v=False 时 mean 为 None，amax 用 max(|max|,|min|)
    # 同上输入：通道 0 amax = max(3, 0) = 3 ; 通道 1 amax = max(20, 0) = 20
    r = solve("fp8_channel_scale", x=[[1.0, 10.0], [3.0, 20.0]],
              scale_max=448.0, smooth_v=False)
    _cmp(r, {
        "mean": [None, None],
        "amax": [3.0, 20.0],
        "scale": [3.0 / 448.0, 20.0 / 448.0],
        "recp_scale": [448.0 / 3.0, 448.0 / 20.0],
        "padded_tokens": 16,
        "head_dim": 2,
    }, "fp8.anchor2")

    # 手算锚点 3：均值分母是 padding 到 16 的长度，不是真实 token 数
    # x=[[4.0]] tokens=1 -> padded=16 ; vals=[4.0]+15 个 0
    #   max=4, min=0, sum=4 ; mean = 4/16 = 0.25（不是 4/1 = 4）
    #   amax = max(|4-0.25|, |0-0.25|) = 3.75
    r = solve("fp8_channel_scale", x=[[4.0]], scale_max=448.0, smooth_v=True)
    _cmp(r, {"mean": [0.25], "amax": [3.75],
             "scale": [3.75 / 448.0], "recp_scale": [448.0 / 3.75],
             "padded_tokens": 16, "head_dim": 1}, "fp8.anchor3")

    # 手算锚点 4：token 数恰好是 16 的倍数时没有 padding
    # x 16 个 token 全为 2.0 -> mean = 32/16 = 2.0 ; min = max = 2.0
    #   amax = max(|2-2|, |2-2|) = 0.0 -> scale = 0.0, recp_scale = inf
    x16 = [[2.0] for _ in range(16)]
    r = solve("fp8_channel_scale", x=x16, scale_max=448.0, smooth_v=True)
    _cmp(r, {"mean": [2.0], "amax": [0.0], "scale": [0.0],
             "recp_scale": [float("inf")], "padded_tokens": 16,
             "head_dim": 1}, "fp8.anchor4")

    # 手算锚点 5：padding 的 0 参与 min 归约 —— 全正数时 min 被拉到 0
    # x=[[5.0, -7.0]] tokens=1 -> padded=16
    # 通道 0: vals=[5.0]+15 个 0 -> max=5, min=0, sum=5 -> mean=5/16=0.3125
    #   amax = max(4.6875, 0.3125) = 4.6875
    # 通道 1: vals=[-7.0]+15 个 0 -> max=0, min=-7, sum=-7 -> mean=-7/16=-0.4375
    #   amax = max(0.4375, 6.5625) = 6.5625
    r = solve("fp8_channel_scale", x=[[5.0, -7.0]], scale_max=448.0,
              smooth_v=True)
    _cmp(r, {"mean": [0.3125, -0.4375], "amax": [4.6875, 6.5625],
             "scale": [4.6875 / 448.0, 6.5625 / 448.0],
             "recp_scale": [448.0 / 4.6875, 448.0 / 6.5625],
             "padded_tokens": 16, "head_dim": 2}, "fp8.anchor5")

    # 手算锚点 6：scale_max 参与 scale 与 recp_scale
    # x=[[3.0]] -> padded=16, mean=3/16=0.1875, amax = max(2.8125, 0.1875)=2.8125
    # scale_max = 240.0 -> scale = 2.8125/240 ; recp = 240/2.8125
    r = solve("fp8_channel_scale", x=[[3.0]], scale_max=240.0, smooth_v=True)
    _cmp(r, {"mean": [0.1875], "amax": [2.8125],
             "scale": [2.8125 / 240.0], "recp_scale": [240.0 / 2.8125],
             "padded_tokens": 16, "head_dim": 1}, "fp8.anchor6")

    # 手算锚点 7：全负通道 —— max/min 初值不能停在 0
    # x=[[-2.0, -6.0]] tokens=2 -> padded=16
    # 通道 0: vals = [-2.0] + 15 个 0 -> max = 0.0（padding 引入）, min = -2.0, sum = -2.0
    #   mean = -2/16 = -0.125 ; amax = max(|0-(-0.125)|, |-2-(-0.125)|) = max(0.125, 1.875) = 1.875
    # 通道 1: vals = [-6.0] + 15 个 0 -> max = 0.0, min = -6.0, sum = -6.0
    #   mean = -0.375 ; amax = max(0.375, 5.625) = 5.625
    r = solve("fp8_channel_scale", x=[[-2.0, -6.0]], scale_max=448.0,
              smooth_v=True)
    _cmp(r, {"mean": [-0.125, -0.375], "amax": [1.875, 5.625],
             "scale": [1.875 / 448.0, 5.625 / 448.0],
             "recp_scale": [448.0 / 1.875, 448.0 / 5.625],
             "padded_tokens": 16, "head_dim": 2}, "fp8.anchor7")

    # 手算锚点 8：16 个 token 全正且无 padding —— min 是真实最小值而非 0
    # x = 16 个 [1.0, 5.0] -> padded = 16（无 padding）
    # 通道 0: max = min = 1.0, sum = 16.0 -> mean = 1.0, amax = 0.0
    # 通道 1: max = min = 5.0, sum = 80.0 -> mean = 5.0, amax = 0.0
    x16 = [[1.0, 5.0] for _ in range(16)]
    r = solve("fp8_channel_scale", x=x16, scale_max=448.0, smooth_v=True)
    _cmp(r, {"mean": [1.0, 5.0], "amax": [0.0, 0.0], "scale": [0.0, 0.0],
             "recp_scale": [float("inf"), float("inf")],
             "padded_tokens": 16, "head_dim": 2}, "fp8.anchor8")

    # 网格对拍：style 与 smooth_v 正交（每种 style 都跑 smooth=True/False）
    rng = _Rng(84012)
    for case in range(160):
        tokens = rng.randint(1, 40)
        head_dim = rng.randint(1, 4)
        style = case % 5
        if style == 0:
            x = [[rng.uniform(-1.0, 1.0) for _ in range(head_dim)]
                 for _ in range(tokens)]
        elif style == 1:
            x = [[rng.uniform(0.1, 3.0) for _ in range(head_dim)]
                 for _ in range(tokens)]
        elif style == 2:
            x = [[rng.uniform(-5.0, -0.1) for _ in range(head_dim)]
                 for _ in range(tokens)]
        elif style == 3:
            x = [[rng.choice([-1.0, 0.0, 1.0, 2.0]) for _ in range(head_dim)]
                 for _ in range(tokens)]
        else:
            # 混合正负，且每通道的 min/max 落在不同 token 上
            x = []
            for t in range(tokens):
                row = []
                for d in range(head_dim):
                    row.append(rng.choice([-4.0, -0.5, 0.5, 4.0]))
                x.append(row)
        smooth = (case // 5) % 2 == 0
        got = solve("fp8_channel_scale", x=x, scale_max=448.0, smooth_v=smooth)
        exp = oracle_fp8_channel(x, 448.0, smooth)
        _cmp(got, exp, "fp8.grid%d" % case)
    print("fp8_channel_scale OK")


def check_transpose_pad_permute(solve):
    # 手算锚点 1：置换索引就是官方注释给的序列
    # dest = (mod//8)*2 + ((mod//2)%4)*4 + (mod%2)
    # 0->0 1->1 2->4 3->5 4->8 5->9 6->12 7->13
    # 8->2 9->3 10->6 11->7 12->10 13->11 14->14 15->15
    expect_perm = [0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15]
    r = solve("transpose_pad_permute", x=[[1.0]], pad_size=16)
    _close(r["permute_index"], expect_perm, tag="perm.anchor1.permute_index")

    # 手算锚点 2：16 个 token 的置换是组内 16 元素的置换
    # x[t] = [t] (head_dim=1)，token=16，pad_size=16 -> padded_tokens=16
    # permuted[0][dest] = padded[0][src]，其中 dest = perm16[src]
    # perm16[0]=0, [1]=1, [2]=4, [3]=5, [4]=8, [5]=9, [6]=12, [7]=13,
    # [8]=2, [9]=3, [10]=6, [11]=7, [12]=10, [13]=11, [14]=14, [15]=15
    # 所以 permuted[0][0]=0, [1]=1, [2]=8, [3]=9, [4]=2, [5]=3, [6]=10, [7]=11,
    # [8]=4, [9]=5, [10]=12, [11]=13, [12]=6, [13]=7, [14]=14, [15]=15
    x16 = [[float(t)] for t in range(16)]
    r = solve("transpose_pad_permute", x=x16, pad_size=16)
    exp_perm = [0.0, 1.0, 8.0, 9.0, 2.0, 3.0, 10.0, 11.0,
                4.0, 5.0, 12.0, 13.0, 6.0, 7.0, 14.0, 15.0]
    _close(r["permuted"], [exp_perm], tag="perm.anchor2.permuted")
    _close(r["padded"], [[float(t) for t in range(16)]], tag="perm.anchor2.padded")
    _close(r["padded_tokens"], 16, tag="perm.anchor2.padded_tokens")

    # 手算锚点 3：padding 到 pad_size 的倍数，padding 位置填 0
    # x=[[1.0],[2.0]] pad_size=64 -> padded_tokens=64
    # padded[0] = [1,2,0,0,...,0]（62 个 0）
    r = solve("transpose_pad_permute", x=[[1.0], [2.0]], pad_size=64)
    _close(r["padded_tokens"], 64, tag="perm.anchor3.padded_tokens")
    _close(r["padded"], [[1.0, 2.0] + [0.0] * 62], tag="perm.anchor3.padded")
    _close(r["permuted"], [[1.0, 2.0] + [0.0] * 62], tag="perm.anchor3.permuted")

    # 手算锚点 4：pad_size=16 时 17 个 token -> 32（补齐到 16 的倍数）
    x17 = [[float(t)] for t in range(17)]
    r = solve("transpose_pad_permute", x=x17, pad_size=16)
    _close(r["padded_tokens"], 32, tag="perm.anchor4.padded_tokens")
    # 第 0 组（token 0..15）被搬到 perm16 位置；token 16 落在第 1 组的位置 0
    _close(r["padded"], [[float(t) for t in range(17)] + [0.0] * 15],
           tag="perm.anchor4.padded")

    # 手算锚点 5：head_dim=2 时转置是真正的转置（行=head_dim，列=token）
    r = solve("transpose_pad_permute", x=[[1.0, 2.0], [3.0, 4.0]],
              pad_size=16)
    _close(r["padded"], [[1.0, 3.0] + [0.0] * 14, [2.0, 4.0] + [0.0] * 14],
           tag="perm.anchor5.padded")
    _close(r["padded_tokens"], 16, tag="perm.anchor5.padded_tokens")

    # 网格对拍
    rng = _Rng(84013)
    for case in range(150):
        tokens = rng.randint(1, 80)
        head_dim = rng.randint(1, 4)
        x = [[rng.uniform(-2.0, 2.0) for _ in range(head_dim)]
             for _ in range(tokens)]
        pad_size = rng.choice([16, 32, 64, 128])
        got = solve("transpose_pad_permute", x=x, pad_size=pad_size)
        exp = oracle_transpose(x, pad_size)
        _cmp(got, exp, "perm.grid%d" % case)
    print("transpose_pad_permute OK")
