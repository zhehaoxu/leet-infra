def _close(a, b, tol=1e-5):
    b = float(b)
    return abs(float(a) - b) <= tol * max(1.0, abs(b))


# 独立重算（对拍口径）：full 按 8 行分块累加、位打包用 divmod 的余数判进位、
# 分组数用 (n-1)//g+1、低秩拆成两个乘积相加、环系数写成 (2 - 2/W)、
# max_bits 从最高位往下扫（与参考解的正向扫描方向相反）。
def _oracle_sign_payload(shapes, bits, group):
    bits = int(bits)
    group = int(group)
    payload = 0
    for rows, cols in shapes:
        n = int(rows) * int(cols)
        if n == 0:
            continue
        q, r = divmod(n * bits, 8)
        packed = q + (1 if r else 0)
        groups = (n - 1) // group + 1 if group >= 1 else 1
        payload += packed + groups * 4
    return payload


def _oracle(shapes, world, method="sign", bytes_per_element=4, bits_per_element=1,
            scale_group_size=0, rank=1, error_feedback=True):
    bpe = int(bytes_per_element)
    full = 0
    elements = 0
    for rows, cols in shapes:                  # 按行分块，避免与参考解同构
        rows, cols = int(rows), int(cols)
        elements += rows * cols
        for start in range(0, rows, 8):
            full += min(8, rows - start) * cols * bpe

    if method == "full":
        payload = full
    elif method == "sign":
        payload = _oracle_sign_payload(shapes, bits_per_element, scale_group_size)
    else:
        payload = 0
        for rows, cols in shapes:
            rows, cols = int(rows), int(cols)
            if int(rank) >= min(rows, cols):
                payload += rows * bpe * cols
            else:
                payload += rows * int(rank) * bpe + cols * int(rank) * bpe

    compressed = payload if payload < full else full
    fallback = payload > full

    w = float(world)
    per_gpu = compressed * (2.0 - 2.0 / w)
    full_comm = full * (2.0 - 2.0 / w)
    saved = (full - compressed) / float(full) if full > 0 else 0.0
    ef = elements * 4 if error_feedback else 0

    max_bits = 0
    if method == "sign" and full > 0:
        for b in range(bpe * 8, 0, -1):
            if _oracle_sign_payload(shapes, b, scale_group_size) <= full:
                max_bits = b
                break
    return full, compressed, per_gpu, full_comm, saved, ef, fallback, max_bits


def _assert_tuple(got, want):
    assert len(got) == 8, got
    assert int(got[0]) == int(want[0]), (got[0], want[0])
    assert int(got[1]) == int(want[1]), (got[1], want[1])
    for k in (2, 3, 4):
        assert _close(got[k], want[k]), (got, want, k)
    assert int(got[5]) == int(want[5]), (got[5], want[5])
    assert bool(got[6]) is bool(want[6]), (got[6], want[6])
    assert int(got[7]) == int(want[7]), (got[7], want[7])


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    sign = ([[4096, 4096]], 8, "sign", 4, 1, 0, 1, True)
    lowrank = ([[1280, 3584]], 8, "lowrank", 2, 1, 0, 8, True)
    fallback = ([[1024, 1024]], 4, "sign", 1, 8, 128, 1, True)
    for c in (sign, lowrank, fallback):
        _assert_tuple(solve(*c), _oracle(*c))

    got = solve(*sign)
    assert int(got[0]) == 67_108_864, got[0]
    assert int(got[1]) == 2_097_156, got[1]           # 2MiB 符号位 + 4 字节整块 scale
    assert _close(got[2], 3_670_023.0), got[2]        # 1.75 * 2097156
    assert _close(got[3], 117_440_512.0), got[3]
    assert _close(got[4], 0.9687499403953552), got[4]
    assert int(got[5]) == 67_108_864 and bool(got[6]) is False, got[5:7]
    assert int(got[7]) == 31, got[7]

    got = solve(*lowrank)
    assert int(got[0]) == 9_175_040, got[0]
    assert int(got[1]) == 77_824, got[1]              # (1280+3584)*8*2
    assert _close(got[2], 136_192.0) and _close(got[3], 16_056_320.0), got[2:4]
    assert _close(got[4], 0.9915178571428571), got[4]
    assert int(got[5]) == 18_350_080 and bool(got[6]) is False, got[5:7]
    assert int(got[7]) == 0, got[7]                   # 非 sign 方法恒 0

    got = solve(*fallback)
    assert int(got[0]) == 1_048_576 and int(got[1]) == 1_048_576, got[:2]
    assert _close(got[2], 1_572_864.0) and _close(got[3], 1_572_864.0), got[2:4]
    assert _close(got[4], 0.0), got[4]
    assert int(got[5]) == 4_194_304 and bool(got[6]) is True, got[5:7]
    assert int(got[7]) == 7, got[7]                   # 第 8 位就超过 fp8 基线


def check_matches_oracle(solve):
    cases = [
        ([[4096, 4096]], 8, "sign", 4, 1, 0, 1, True),
        ([[4096, 4096]], 8, "sign", 4, 1, 1024, 1, True),
        ([[1280, 3584]], 8, "lowrank", 2, 1, 0, 8, True),
        ([[1024, 1024]], 4, "sign", 1, 8, 128, 1, True),
        ([[2048, 512], [512, 2048], [512, 512]], 16, "sign", 4, 2, 256, 1, True),
        ([[1024, 1024]], 8, "lowrank", 4, 1, 0, 512, True),        # 准满秩：压缩反而更贵
        ([[1024, 1024]], 8, "lowrank", 4, 1, 0, 1024, True),       # rank==min -> 按满秩计
        ([[8, 8]], 2, "full", 4, 1, 0, 1, True),
        ([[777, 129]], 3, "sign", 4, 16, 64, 1, False),
        ([[1, 1]], 8, "sign", 4, 1, 0, 1, True),                  # scale 就比原值贵
        ([[1024, 1024]], 64, "sign", 2, 3, 512, 1, True),
        ([[2048, 4096], [4096, 2048]], 8, "lowrank", 2, 1, 0, 16, True),
        ([[0, 64], [64, 0]], 4, "sign", 4, 1, 0, 1, True),        # 空矩阵跳过
        ([[96, 32], [32, 96]], 128, "sign", 4, 4, 32, 1, True),
    ]
    for c in cases:
        _assert_tuple(solve(*c), _oracle(*c))


def check_world_one_and_ring_coefficient(solve):
    single = solve([[4096, 4096]], 1, "sign", 4, 1, 0, 1, True)
    assert _close(single[2], 0.0) and _close(single[3], 0.0), single[2:4]
    assert int(single[0]) == 67_108_864 and int(single[1]) == 2_097_156, single[:2]
    assert _close(single[4], 0.9687499403953552), single[4]
    # 每 rank 搬运量 = (2 - 2/W) * payload：W 翻倍时系数从 1.5 -> 1.75，不是线性翻倍
    s2, s4, s8 = (solve([[4096, 4096]], w, "sign", 4, 1, 0, 1, True)[2] for w in (2, 4, 8))
    payload = 2_097_156
    assert _close(s2, payload) and _close(s4, 1.5 * payload) and _close(s8, 1.75 * payload), (s2, s4, s8)
    assert s8 < 2.0 * s4 < 4.0 * s2, (s2, s4, s8)


def check_full_method_is_the_baseline(solve):
    shapes = [[2048, 512], [512, 2048], [512, 512]]     # 2,359,296 个元素
    full = solve(shapes, 8, "full", 4, 1, 0, 1, True)
    assert int(full[0]) == 9_437_184 and int(full[1]) == 9_437_184, full[:2]
    assert _close(full[2], full[3]) and _close(full[4], 0.0), full[2:5]
    assert bool(full[6]) is False and int(full[7]) == 0, full[6:8]
    assert int(full[5]) == 9_437_184, full[5]           # 残差累加器恒 fp32
    # 未压缩基线随 bytes_per_element 线性缩放；sign 载荷只看位数，不受 bpe 影响
    fp16 = solve(shapes, 8, "full", 2, 1, 0, 1, True)
    assert int(fp16[0]) == 4_718_592 and _close(fp16[2], full[2] / 2.0), fp16[:3]
    sign = solve(shapes, 8, "sign", 2, 1, 0, 1, True)
    assert int(sign[1]) == 294_924 and int(sign[0]) == int(fp16[0]), sign[:2]
    assert _close(sign[2], 516_117.0) and bool(sign[6]) is False, sign[2:3] + (sign[6],)
    # 同一份 sign 配置在 fp32 基线下省得更多（分母变大）
    sign32 = solve(shapes, 8, "sign", 4, 1, 0, 1, True)
    assert int(sign32[1]) == int(sign[1]) and sign32[4] > sign[4], (sign32[1], sign32[4], sign[4])


def check_fallback_and_max_bits(solve):
    # 同一份配置，fp32 基线下压缩有效、fp8 基线下反而更贵 -> 回退
    shapes = [[1024, 1024]]
    warm = solve(shapes, 4, "sign", 4, 8, 128, 1, True)
    poor = solve(shapes, 4, "sign", 1, 8, 128, 1, True)
    assert bool(warm[6]) is False and int(warm[1]) == 1_081_344, warm[:2]
    assert bool(poor[6]) is True and int(poor[1]) == int(poor[0]), poor[:2]
    assert _close(poor[4], 0.0), poor[4]
    # max_bits 是切换点：该位数可行，再多 1 位就回退（bpe=4 -> 上限 32 位）
    for bpe, bits in [(4, 31), (2, 15), (1, 7)]:
        mb = solve(shapes, 4, "sign", bpe, 1, 128, 1, True)[7]
        assert mb == bpe * 8 - 1, (bpe, mb)
        assert bool(solve(shapes, 4, "sign", bpe, mb, 128, 1, True)[6]) is False, mb
        assert bool(solve(shapes, 4, "sign", bpe, mb + 1, 128, 1, True)[6]) is True, mb
        assert mb <= bpe * 8, (bpe, mb)
    # 单元素张量：一个 fp32 scale(4B) 就抵得上整个未压缩量，永远压不动
    tiny = solve([[1, 1]], 8, "sign", 4, 1, 0, 1, True)
    assert int(tiny[0]) == 4 and int(tiny[1]) == 4 and int(tiny[7]) == 0, tiny
    assert bool(tiny[6]) is True, tiny[6]


def check_lowrank_rank_boundary(solve):
    shapes = [[1024, 1024]]
    full_bytes = 4_194_304
    # rank == min(rows, cols)：满秩，按原矩阵计，恰好等于 full（不算回退，因为 payload == full）
    sat = solve(shapes, 8, "lowrank", 4, 1, 0, 1024, True)
    assert int(sat[1]) == full_bytes and bool(sat[6]) is False, sat[:2] + (sat[6],)
    assert _close(sat[4], 0.0), sat[4]
    # rank == min - 1：P+Q 两块 ((1024+1024)*1023*4 = 8,379,904) 比原矩阵还大 -> 回退
    near = solve(shapes, 8, "lowrank", 4, 1, 0, 1023, True)
    assert int(near[1]) == full_bytes and bool(near[6]) is True, near[:2] + (near[6],)
    # 瘦长矩阵对秩更宽容：rows=8, cols=4096 时 rank=8 就已经满秩
    thin_full = solve([[8, 4096]], 8, "full", 4, 1, 0, 1, True)
    for r in (8, 16, 64):
        got = solve([[8, 4096]], 8, "lowrank", 4, 1, 0, r, True)
        assert int(got[1]) == int(thin_full[1]) and bool(got[6]) is False, (r, got[:2])
    # rank=1 时载荷 = (rows+cols)*bpe，与矩阵大小成线性而非平方
    one = solve([[1024, 1024]], 8, "lowrank", 4, 1, 0, 1, True)
    assert int(one[1]) == 2048 * 4, one[1]
    assert _close(one[4], 1.0 - 8192 / full_bytes), one[4]


def check_group_scale_and_empty_shapes(solve):
    shapes = [[1024, 1024], [512, 512]]
    base = solve(shapes, 8, "sign", 4, 1, 0, 1, True)
    # g=0：每块张量各付一个 scale（这里 2 个 -> +8 字节）；g=numels 时与之等价
    per_tensor = solve(shapes, 8, "sign", 4, 1, 1_048_576, 1, True)
    assert int(base[1]) == 163_840 + 8, base[1]       # 2^20/8 + 2^18/8 位打包 + 两个 scale
    assert int(per_tensor[1]) == int(base[1]), (base[1], per_tensor[1])
    # 分组越细，scale 开销单调上升；细到每元素一个 scale 时已经超过基线 -> 被回退钳住
    prev = int(base[1])
    for g in (65536, 4096, 256, 16):
        cur = int(solve(shapes, 8, "sign", 4, 1, g, 1, True)[1])
        assert cur >= prev, (g, cur, prev)
        prev = cur
    assert prev == 163_840 + (1_048_576 + 262_144) // 16 * 4, prev
    coarse = solve(shapes, 8, "sign", 4, 1, 1, 1, True)
    assert int(coarse[1]) == int(coarse[0]) == 5_242_880, coarse[:2]
    assert bool(coarse[6]) is True and _close(coarse[4], 0.0), coarse[4:7]
    assert int(coarse[7]) == 0, coarse[7]             # 光 scale 就吃满基线，没有可行位数
    assert int(solve(shapes, 8, "sign", 4, 1, 16, 1, True)[7]) == 30, "g=16 时还能压到 30 位"
    # 全空形状清单：一切归零，max_bits 也记 0
    empty = solve([], 8, "sign", 4, 1, 0, 1, True)
    assert int(empty[0]) == 0 and int(empty[1]) == 0 and int(empty[7]) == 0, empty
    assert _close(empty[2], 0.0) and _close(empty[4], 0.0) and int(empty[5]) == 0, empty
    # error_feedback=False 只影响残差字节
    off = solve(shapes, 8, "sign", 4, 1, 0, 1, False)
    assert int(off[5]) == 0 and int(off[0]) == int(base[0]) and int(off[1]) == int(base[1]), off[5]


def check_invariants_over_a_sweep(solve):
    # 不变量扫描：压缩量不会超过未压缩基线，通信字节与 saved_ratio 自洽
    grid = []
    for shapes in ([[256, 256]], [[1024, 128], [128, 1024]], [[512, 512], [512, 1024]]):
        for world in (1, 2, 8, 32):
            for method, bpe, bits, group, rank in [("sign", 4, 1, 0, 1), ("sign", 4, 8, 256, 1),
                                                   ("sign", 1, 2, 64, 1), ("lowrank", 4, 1, 0, 4),
                                                   ("lowrank", 2, 1, 0, 128), ("full", 4, 1, 0, 1)]:
                grid.append((shapes, world, method, bpe, bits, group, rank, True))
    for c in grid:
        got = solve(*c)
        _assert_tuple(got, _oracle(*c))
        full, compressed, per_gpu, full_comm, saved, ef, _fallback, _mb = got
        elements = sum(int(r) * int(cc) for r, cc in c[0])
        assert 0 <= compressed <= full, c
        assert per_gpu <= full_comm + 1e-9, (c, per_gpu, full_comm)
        assert _close(saved, (full - compressed) / float(full) if full else 0.0), (c, saved)
        assert _close(per_gpu, compressed * (2.0 - 2.0 / c[1])), (c, per_gpu)
        assert _close(full_comm, full * (2.0 - 2.0 / c[1])), (c, full_comm)
        assert int(ef) == elements * 4, (c, ef)
        assert int(full) == elements * c[3], (c, full)


check_example.example = True
