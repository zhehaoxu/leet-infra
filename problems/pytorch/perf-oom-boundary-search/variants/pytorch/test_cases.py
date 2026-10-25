# 独立重算（对拍口径）：两条边界都用**纯整数二分**（不碰二次公式、不碰浮点），
# 激活字节把 m 提到括号外、headroom 一路减下来、world_size 用「商 + 余数进位」、
# global_batch_fits 用取模判据（与参考解的乘法判据结构不同）。
def _oracle(hbm_bytes, reserve_bytes, cuda_overhead_bytes, param_bytes, grad_bytes, optim_bytes,
            act_per_token, act_per_token_sq, micro_batch, seq, global_batch, accum_steps):
    static = sum([param_bytes, grad_bytes, optim_bytes])
    usable = hbm_bytes - reserve_bytes
    ap = act_per_token
    apq = act_per_token_sq
    m = micro_batch
    s = seq

    def peak(mm, ss):
        return cuda_overhead_bytes + static + mm * (ss * ap + ss * (ss * apq))

    def fits(mm, ss):
        return peak(mm, ss) <= usable

    peak_bytes = peak(m, s)
    headroom = usable - cuda_overhead_bytes - static - m * (s * ap + s * (s * apq))
    feasible = peak_bytes <= usable
    budget = usable - cuda_overhead_bytes - static

    if budget < 0:
        max_m, max_s = 0, 0
    else:
        lo, hi = 0, budget // (s * ap) + 1            # 只算线性项就是 m 的合法上界
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if fits(mid, s):
                lo = mid
            else:
                hi = mid
        max_m = lo if lo >= 1 else 0
        lo, hi = 0, budget // (m * ap) + 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if fits(m, mid):
                lo = mid
            else:
                hi = mid
        max_s = lo if lo >= 1 else 0

    per_rank = accum_steps * m
    world = global_batch // per_rank + (1 if global_batch % per_rank else 0)
    return (static, peak_bytes, headroom, feasible, max_m, max_s, world,
            bool(feasible and global_batch % per_rank == 0))


def _assert_tuple(got, want):
    assert len(got) == 8, got
    for k in (0, 1, 2, 4, 5, 6):
        assert int(got[k]) == int(want[k]), (got, want, k)
    assert bool(got[3]) is bool(want[3]), (got[3], want[3])
    assert bool(got[7]) is bool(want[7]), (got[7], want[7])


def _peak(overhead, static, ap, apq, m, s):
    """把模型公式原样代一遍，作为「到底放不放得下」的第三方裁判。"""
    return overhead + static + m * s * ap + m * s * s * apq


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    a100 = (80_000_000_000, 2_000_000_000, 1_500_000_000,
            2_600_000_000, 2_600_000_000, 15_600_000_000)
    fits_case = a100 + (1_500_000, 128, 4, 2048, 512, 8)
    oom_case = a100 + (1_500_000, 128, 8, 8192, 512, 8)
    exact = (20_000, 0, 0, 0, 0, 0, 1000, 0, 2, 10, 40, 4)
    for c in (fits_case, oom_case, exact):
        _assert_tuple(solve(*c), _oracle(*c))

    got = solve(*fits_case)
    assert int(got[0]) == 20_800_000_000, got[0]                    # 2+2+12 字节/参数口径
    assert int(got[1]) == 36_735_483_648, got[1]                    # 12.288GB 线性 + 2.147GB 二次
    assert int(got[2]) == 41_264_516_352 and bool(got[3]) is True, got[2:4]
    assert int(got[4]) == 15 and int(got[5]) == 6103, got[4:6]
    assert int(got[6]) == 16 and bool(got[7]) is True, got[6:8]

    got = solve(*oom_case)
    assert int(got[1]) == 189_323_476_736, got[1]
    assert int(got[2]) == -111_323_476_736 and bool(got[3]) is False, got[2:4]
    assert int(got[4]) == 2 and int(got[5]) == 3560, got[4:6]       # 还有救：缩到 2 个 micro batch
    assert int(got[6]) == 8 and bool(got[7]) is False, got[6:8]     # 整除也救不了 OOM

    got = solve(*exact)
    assert int(got[1]) == 20_000 and int(got[2]) == 0 and bool(got[3]) is True, got[1:4]
    assert int(got[4]) == 2 and int(got[5]) == 10, got[4:6]         # 刚好用满：闭区间
    assert int(got[6]) == 5 and bool(got[7]) is True, got[6:8]


def check_matches_oracle(solve):
    big = (80_000_000_000, 2_000_000_000, 1_500_000_000,
           2_600_000_000, 2_600_000_000, 15_600_000_000)
    cases = [
        big + (1_500_000, 128, 4, 2048, 512, 8),
        big + (1_500_000, 128, 8, 8192, 512, 8),
        big + (1_500_000, 0, 2, 1024, 100, 5),                      # flash attention：只剩线性项
        big + (250_000, 7, 3, 3333, 999, 7),                        # 小系数、处处带余数
        (20_000, 0, 0, 0, 0, 0, 1000, 0, 2, 10, 40, 4),
        (19_999, 0, 0, 0, 0, 0, 1000, 0, 2, 10, 40, 4),
        (20_001, 0, 0, 0, 0, 0, 1000, 0, 2, 10, 41, 4),             # 41 不整除 A*m
        (32_000_000_000, 30_000_000_000, 1_000_000_000,
         1_000_000_000, 1_000_000_000, 1_000_000_000, 1_000_000, 100, 1, 128, 64, 2),
        (40_000_000_000, 2_000_000_000, 1_000_000_000,
         1_300_000_000, 1_300_000_000, 7_800_000_000, 800_000, 0, 16, 4096, 256, 4),
        (1_000_000_000, 0, 0, 500_000_000, 0, 0, 1, 1, 1, 1000, 8, 1),   # 二次项压倒一切
        (8_000_000_000_000, 0, 0, 0, 0, 0, 1, 0, 1_000_000, 1_000_000, 1_000, 1),
        (140_000_000_000, 5_000_000_000, 2_000_000_000,
         13_200_000_000, 13_200_000_000, 79_200_000_000, 4_000_000, 512, 1, 512, 1024, 16),
    ]
    for c in cases:
        _assert_tuple(solve(*c), _oracle(*c))


def check_inclusive_boundary_and_headroom(solve):
    for hbm, want_fit, want_m, want_s in [(20_000, True, 2, 10),
                                          (19_999, False, 1, 9),
                                          (20_001, True, 2, 10),
                                          (40_000, True, 4, 20)]:
        got = solve(hbm, 0, 0, 0, 0, 0, 1000, 0, 2, 10, 40, 4)
        assert int(got[1]) == 20_000, (hbm, got[1])                 # 峰值只由 m,s 决定
        assert bool(got[3]) is want_fit, (hbm, got[3])
        assert int(got[2]) == hbm - 20_000, (hbm, got[2])           # headroom 可为负、不 clamp
        assert int(got[4]) == want_m and int(got[5]) == want_s, (hbm, got[4:6])
    # 相等那一档：差 1 字节就是两个世界
    tight = solve(20_000, 0, 0, 0, 0, 0, 1000, 0, 2, 10, 40, 4)
    assert int(tight[2]) == 0 and bool(tight[3]) is True and bool(tight[7]) is True, tight[2:4]
    # 预留是从额度里直接扣的（不进峰值），cuda_overhead 却算进峰值
    reserved = solve(80_000_000_000, 8_000_000_000, 1_000_000_000, 1_000_000_000, 0, 0,
                     1_000_000, 0, 1, 10, 10, 1)
    plain = solve(80_000_000_000, 0, 1_000_000_000, 1_000_000_000, 0, 0,
                  1_000_000, 0, 1, 10, 10, 1)
    assert int(reserved[1]) == int(plain[1]), (reserved[1], plain[1])
    assert int(reserved[2]) == int(plain[2]) - 8_000_000_000, (reserved[2], plain[2])
    assert int(reserved[0]) == 1_000_000_000, reserved[0]


def check_max_micro_batch_is_an_integer_staircase(solve):
    for k in range(1, 6):
        hbm = k * 10 * 1000                       # 每个 micro batch 花 10 位置 * 1000 字节
        got = solve(hbm, 0, 0, 0, 0, 0, 1000, 0, 1, 10, 1, 1)
        assert int(got[4]) == k, (hbm, got[4])
        assert bool(got[3]) is True, got[3]
        below = solve(hbm - 1, 0, 0, 0, 0, 0, 1000, 0, 1, 10, 1, 1)
        assert int(below[4]) == k - 1, (hbm, below[4])              # 少 1 字节就掉一级
    # 二次项也在每份成本里：s 变大时成本是 线性 + 平方 的混合
    for s in (4, 16, 64):
        got = solve(100_000_000, 0, 0, 0, 0, 0, 100_000, 37, 3, s, 12, 4)
        cost = s * 100_000 + s * s * 37
        assert int(got[4]) == 100_000_000 // cost, (s, got[4])
        assert _peak(0, 0, 100_000, 37, int(got[4]), s) <= 100_000_000, s
        assert _peak(0, 0, 100_000, 37, int(got[4]) + 1, s) > 100_000_000, s


def check_quadratic_seq_boundary_is_the_hard_part(solve):
    head = (80_000_000_000, 2_000_000_000, 1_500_000_000,
            2_600_000_000, 2_600_000_000, 15_600_000_000, 1_500_000)
    usable, overhead, static, budget = 78_000_000_000, 1_500_000_000, 20_800_000_000, 55_700_000_000
    for m in (1, 4, 32):
        got = solve(*head, 128, m, 2048, 512, 8)
        top = int(got[5])
        assert top >= 1, (m, top)
        assert _peak(overhead, static, 1_500_000, 128, m, top) <= usable, (m, top)
        assert _peak(overhead, static, 1_500_000, 128, m, top + 1) > usable, (m, top)
        assert top <= budget // (m * 1_500_000), (m, top)           # 线性解永远是不松的上界
    assert solve(*head, 128, 4, 2048, 512, 8)[5] == 6103           # 二次方程正根的 floor
    # 关掉二次项（flash attention）后同一份预算能开更长：线性 floor 直接给答案
    lin = solve(*head, 0, 4, 2048, 512, 8)
    quad = solve(*head, 128, 4, 2048, 512, 8)
    assert int(lin[5]) == 55_700_000_000 // (4 * 1_500_000), lin[5]  # 9283
    assert int(lin[5]) > int(quad[5]), (lin[5], quad[5])
    # 二次项系数变大 -> 上限远小于按线性算出来的值
    heavy = solve(*head, 2048, 4, 2048, 512, 8)
    assert int(heavy[5]) < int(quad[5]), (heavy[5], quad[5])
    assert _peak(overhead, static, 1_500_000, 2048, 4, int(heavy[5])) <= usable


def check_no_budget_and_zero_static(solve):
    # 预留吃掉大半显存：连固定开销都放不下 -> 两条边界都是 0
    got = solve(32_000_000_000, 30_000_000_000, 1_000_000_000,
                1_000_000_000, 1_000_000_000, 1_000_000_000, 1_000_000, 100, 1, 128, 64, 2)
    assert bool(got[3]) is False and int(got[2]) < 0, got[2:4]
    assert int(got[4]) == 0 and int(got[5]) == 0, got[4:6]
    assert int(got[6]) == 32 and bool(got[7]) is False, got[6:8]    # world_size 照样算得出
    # 预算刚好够固定开销时，至少能开 1 个 micro batch / 1 个 token
    ok = solve(1_000_000, 0, 0, 0, 0, 0, 500_000, 100, 1, 1, 4, 2)
    assert bool(ok[3]) is True and int(ok[4]) >= 1 and int(ok[5]) >= 1, ok[3:6]
    assert int(ok[2]) == 1_000_000 - 500_100, ok[2]
    # 每 micro batch 只花 1 字节时，边界等于全部预算（退化到「按字节数」）
    thin = solve(2, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1)
    assert int(thin[1]) == 1 and bool(thin[3]) is True and int(thin[4]) == 2, thin[:2] + (thin[4],)
    assert int(solve(1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1)[4]) == 1
    # 只留参数（推理或把梯度/优化器 offload 掉）：static 只剩 param
    infer = solve(80_000_000_000, 0, 0, 13_200_000_000, 0, 0, 1, 0, 1, 1, 1, 1)
    assert int(infer[0]) == 13_200_000_000, infer[0]
    assert int(infer[4]) == 80_000_000_000 - 13_200_000_000, infer[4]


def check_world_size_and_global_batch_divisibility(solve):
    base = (80_000_000_000, 2_000_000_000, 1_500_000_000,
            2_600_000_000, 2_600_000_000, 15_600_000_000, 1_500_000, 128)
    for g, accum, m, want_world, want_fits in [
        (512, 8, 4, 16, True),
        (500, 8, 4, 16, False),        # ceil 之后 16*32=512 != 500
        (32, 8, 4, 1, True),
        (31, 8, 4, 1, False),          # 一个 rank 也凑不出 31 条
        (64, 8, 4, 2, True),
        (120, 8, 4, 4, False),         # 120 能被 8*15 整除，但 world 只看给定的 m=4
        (1_000_000, 1, 1, 1_000_000, True),
    ]:
        got = solve(*base, m, 256, g, accum)
        assert bool(got[3]) is True, (g, m, got[3])
        assert int(got[6]) == want_world, (g, accum, m, got[6], want_world)
        assert bool(got[7]) is want_fits, (g, accum, m, got[7], want_fits)
    # 显存不可行时，整除也白搭
    oom = solve(*base, 8, 8192, 512, 8)
    assert int(oom[6]) == 8 and bool(oom[3]) is False and bool(oom[7]) is False, oom[6:8]


def check_headroom_and_peak_invariants(solve):
    configs = [
        (80_000_000_000, 2_000_000_000, 1_500_000_000, 2_600_000_000, 2_600_000_000,
         15_600_000_000, 1_500_000, 128, 4, 2048, 512, 8),
        (40_000_000_000, 1_000_000_000, 500_000_000, 900_000_000, 900_000_000,
         5_400_000_000, 300_000, 2048, 6, 512, 200, 3),
        (6_000_000, 1_000_000, 100_000, 200_000, 200_000, 900_000, 1234, 7, 3, 99, 24, 6),
        (9_000_000, 8_000_000, 1_000_000, 1_000_000, 1_000_000, 1_000_000, 500_000, 50, 2, 64, 16, 2),
    ]
    for c in configs:
        hbm, reserve, overhead, param, grad, optim, ap, apq, m, s, g, a = c
        static, peak_bytes, headroom, feasible, max_m, max_s, world, fits_all = solve(*c)
        usable = hbm - reserve
        assert int(static) == param + grad + optim, c
        assert int(peak_bytes) == _peak(overhead, int(static), ap, apq, m, s), peak_bytes
        assert int(headroom) == usable - int(peak_bytes), (headroom, peak_bytes)
        assert bool(feasible) is (int(headroom) >= 0), (feasible, headroom)
        assert int(world) >= 1 and int(world) * a * m >= g, (world, g)
        assert bool(fits_all) is (bool(feasible) and int(world) * a * m == g), fits_all
        # 两条边界各自代回公式验一次：peak 对 m 线性、对 s 是二次，不能互换位置
        if int(max_m) >= 1:
            assert _peak(overhead, int(static), ap, apq, int(max_m), s) <= usable, (c, max_m)
            assert _peak(overhead, int(static), ap, apq, int(max_m) + 1, s) > usable, (c, max_m)
        else:
            assert _peak(overhead, int(static), ap, apq, 1, s) > usable, (c, max_m)
        if int(max_s) >= 1:
            assert _peak(overhead, int(static), ap, apq, m, int(max_s)) <= usable, (c, max_s)
            assert _peak(overhead, int(static), ap, apq, m, int(max_s) + 1) > usable, (c, max_s)
        else:
            assert _peak(overhead, int(static), ap, apq, m, 1) > usable, (c, max_s)


check_example.example = True
