def _close(a, b, tol=1e-5):
    b = float(b)
    return abs(float(a) - b) <= tol * max(1.0, abs(b))


# 独立重算（对拍口径）：tokens/s 走「先除以 batch 得到每 token 时间、再取 max 求倒数」，
# 权重占比写成 1/(1 + batch*K/W)，上限写成 1/max(每 token 算力时间, 每 token KV 时间)，
# 拐点 batch 用 1,2,4,... 指数扩张后二分（不用闭式除法），SLO 最小 batch 从 1 起线性扫描。
def _oracle(n_params, layers, kv_heads, head_dim, context_tokens, batch,
            peak_flops, bandwidth, weight_bytes=2, kv_bytes=2,
            target_tokens_per_s=0.0, max_batch=100000):
    weight = float(n_params) * float(weight_bytes)
    kv = float(2 * layers * kv_heads * head_dim * kv_bytes) * float(context_tokens)
    fpt = 2 * int(n_params)

    def tps_total(b):
        return b / max((weight + b * kv) / bandwidth, b * fpt / peak_flops)

    def tps_per_token(b):                       # 与 tps_total 结构不同的另一条推导
        return 1.0 / max((weight + b * kv) / (b * bandwidth), fpt / peak_flops)

    def compute_bound(b):
        return b * fpt * bandwidth >= (weight + b * kv) * peak_flops

    tokens_per_s = tps_per_token(batch)
    step_ms = 1000.0 * (batch / tokens_per_s)
    weight_fraction = 1.0 / (1.0 + batch * kv / weight)
    ceiling = 1.0 / max(fpt / peak_flops, (kv / bandwidth) if kv > 0.0 else 0.0)

    balance = -1
    if compute_bound(1):
        balance = 1
    else:
        lo, hi = 1, 2
        while hi <= 2 ** 62 and not compute_bound(hi):
            lo, hi = hi, hi * 2
        if compute_bound(hi):
            while lo + 1 < hi:
                mid = (lo + hi) // 2
                if compute_bound(mid):
                    hi = mid
                else:
                    lo = mid
            balance = hi

    min_batch = -1
    for b in range(1, int(max_batch) + 1):
        if tps_total(b) >= float(target_tokens_per_s):
            min_batch = b
            break

    return (tokens_per_s, compute_bound(batch) is False, step_ms, ceiling,
            balance, min_batch, weight_fraction, fpt)


def _assert_tuple(got, want):
    assert len(got) == 8, got
    assert _close(got[0], want[0]), (got[0], want[0])
    assert bool(got[1]) is bool(want[1]), (got[1], want[1])
    assert _close(got[2], want[2]), (got[2], want[2])
    assert _close(got[3], want[3]), (got[3], want[3])
    assert int(got[4]) == int(want[4]), (got[4], want[4])
    assert int(got[5]) == int(want[5]), (got[5], want[5])
    assert _close(got[6], want[6]), (got[6], want[6])
    assert int(got[7]) == int(want[7]), (got[7], want[7])


def _run(solve, args, kwargs=None):
    kwargs = kwargs or {}
    return solve(*args, **kwargs), _oracle(*args, **kwargs)


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    llama = (6_600_000_000, 32, 8, 128, 2048, 32, 312e12, 2e12)
    got, want = _run(solve, llama, dict(target_tokens_per_s=2000.0))
    _assert_tuple(got, want)
    assert _close(got[0], 2937.135939063217), got[0]
    assert bool(got[1]) is True and _close(got[2], 10.894967296), got[1:3]
    assert _close(got[3], 7450.580596923828), got[3]            # 2e12 / 268,435,456
    assert int(got[4]) == -1, got[4]                            # KV 读永远压不住
    assert int(got[5]) == 19 and _close(got[6], 0.6057842874317886), got[5:7]
    assert int(got[7]) == 13_200_000_000, got[7]

    zero_ctx = (6_600_000_000, 32, 8, 128, 0, 32, 312e12, 2e12)
    got, want = _run(solve, zero_ctx, dict(target_tokens_per_s=2000.0))
    _assert_tuple(got, want)
    assert _close(got[0], 4848.484848484849) and bool(got[1]) is True, got[:2]
    assert _close(got[2], 6.6) and _close(got[3], 23636.363636363636), got[2:4]
    assert int(got[4]) == 156 and int(got[5]) == 14, got[4:6]   # 拐点 = peak*wb/(2*bw)
    assert _close(got[6], 1.0), got[6]

    edge = (1000, 2, 2, 8, 0, 1, 1e9, 1e9)
    got, want = _run(solve, edge)
    _assert_tuple(got, want)
    assert _close(got[0], 500000.0) and bool(got[1]) is False, got[:2]   # 相等记 compute-bound
    assert _close(got[2], 0.002) and _close(got[3], 500000.0), got[2:4]
    assert int(got[4]) == 1 and int(got[5]) == 1 and _close(got[6], 1.0), got[4:7]
    assert int(got[7]) == 2000, got[7]


def check_matches_oracle(solve):
    cases = [
        ((6_600_000_000, 32, 8, 128, 2048, 32, 312e12, 2e12), {"target_tokens_per_s": 2000.0, "max_batch": 1000}),
        ((6_600_000_000, 32, 8, 128, 0, 32, 312e12, 2e12), {"target_tokens_per_s": 2000.0, "max_batch": 1000}),
        ((1000, 2, 2, 8, 0, 1, 1e9, 1e9), {}),
        ((1000, 2, 2, 8, 64, 4, 1e9, 2e12), {"target_tokens_per_s": 400000.0, "max_batch": 64}),
        ((6_600_000_000, 32, 8, 128, 512, 1024, 312e12, 2e12), {"target_tokens_per_s": 20000.0, "max_batch": 2000}),
        ((6_600_000_000, 32, 32, 128, 128, 1, 312e12, 2e12), {"weight_bytes": 1, "kv_bytes": 1,
                                                              "target_tokens_per_s": 100.0, "max_batch": 100}),
        ((7_000_000_000, 32, 8, 128, 4096, 1, 989e12, 3.35e12), {"target_tokens_per_s": 50.0, "max_batch": 100}),
        ((1_000_000_000, 24, 4, 64, 3000, 777, 400e12, 1.6e12), {"max_batch": 5000}),
        ((100_000_000, 4, 4, 32, 16, 8, 1e12, 1e11), {"target_tokens_per_s": 1e9, "max_batch": 100}),
        ((3_000_000_000, 28, 7, 128, 1, 2, 312e12, 2e12), {"weight_bytes": 4, "kv_bytes": 4,
                                                           "target_tokens_per_s": 10.0, "max_batch": 300}),
        ((500_000_000, 2, 1, 8, 2048, 100000, 1e12, 1e11), {"max_batch": 200}),
        ((6_600_000_000, 32, 8, 128, 2048, 1, 312e12, 2e12), {"target_tokens_per_s": 30000.0, "max_batch": 1000}),
    ]
    for args, kwargs in cases:
        got, want = _run(solve, args, kwargs)
        _assert_tuple(got, want)


def check_batch_scaling_saturates_at_the_ceiling(solve):
    llama = dict(n_params=6_600_000_000, layers=32, kv_heads=8, head_dim=128,
                 context_tokens=512, peak_flops=312e12, bandwidth=2e12)
    prev = 0.0
    for b in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
        got = solve(batch=b, **llama)
        assert got[0] >= prev - 1e-9, (b, got[0], prev)         # 加 batch 只会更快
        assert got[0] <= got[3] + 1e-6, (b, got[0], got[3])     # 从不超过上限
        assert bool(got[1]) is True, (b, got[1])                # 512 上下文下还没到拐点
        prev = got[0]
    # 拐点之后进入平台：tokens/s 恒等于算力上限，step_time 随 batch 线性增长
    knee = solve(batch=1, **llama)[4]
    assert knee == 755, knee
    plateau = [solve(batch=b, **llama) for b in (knee, 4096, 16384)]
    for b, got in zip((knee, 4096, 16384), plateau):
        assert bool(got[1]) is False, (b, got[1])
        assert _close(got[0], got[3]), (b, got[0], got[3])      # = peak/(2N) = 23636.36...
    assert _close(plateau[1][2] / plateau[0][2], 4096.0 / knee), (plateau[0][2], plateau[1][2])
    # batch 极小时权重占绝对主导：tokens/s 约等于 batch*bandwidth/weight（线性提速）
    small = solve(batch=1, **llama)
    assert _close(small[0], 2e12 / (6_600_000_000 * 2.0), tol=0.05) or small[0] < 200, small[0]
    assert small[6] > 0.9, small[6]


def check_balance_knee_is_independent_of_model_size(solve):
    # context_tokens=0 时 K=0，拐点 = peak*weight_bytes/(2*bandwidth)，与 n_params 无关
    for n in (100_000_000, 1_000_000_000, 6_600_000_000, 70_000_000_000):
        knee = solve(n, 32, 8, 128, 0, 1, 312e12, 2e12)[4]
        assert knee == 156, (n, knee)
    # 权重字节减半（fp16 -> fp8）拐点也减半；带宽翻倍同样减半
    assert solve(6_600_000_000, 32, 8, 128, 0, 1, 312e12, 2e12, weight_bytes=1)[4] == 78
    assert solve(6_600_000_000, 32, 8, 128, 0, 1, 312e12, 4e12)[4] == 78
    # 拐点判据：K 每翻倍一次，拐点就往后推，直到 KV 读彻底压不住 -> -1
    prev = 0
    for ctx in (0, 16, 64, 256, 1024):
        knee = solve(6_600_000_000, 32, 8, 128, ctx, 1, 312e12, 2e12)[4]
        assert knee == -1 or knee > prev, (ctx, knee, prev)
        prev = max(prev, knee)
    assert solve(6_600_000_000, 32, 8, 128, 1024, 1, 312e12, 2e12)[4] == -1
    # 拐点两侧各差一个 batch：正好是「memory -> compute」的切换
    got_lo = solve(6_600_000_000, 32, 8, 128, 512, 754, 312e12, 2e12)
    got_hi = solve(6_600_000_000, 32, 8, 128, 512, 755, 312e12, 2e12)
    assert bool(got_lo[1]) is True and bool(got_hi[1]) is False, (got_lo[1], got_hi[1])
    assert got_hi[0] > got_lo[0], (got_lo[0], got_hi[0])


def check_kv_traffic_halves_the_ceiling(solve):
    args = dict(n_params=6_600_000_000, layers=32, kv_heads=8, head_dim=128,
                peak_flops=312e12, bandwidth=2e12, batch=1)
    base = solve(context_tokens=2048, **args)
    double = solve(context_tokens=4096, **args)
    # KV 字节与上下文长度成正比：ceiling（= bw/K）正好减半
    assert _close(base[3], double[3] * 2.0), (base[3], double[3])
    # 但单步耗时几乎不变：权重读量远大于 KV，翻倍上下文只让 step_ms 涨一点点
    ratio = double[2] / base[2]
    assert 1.0 < ratio < 1.05, (base[2], double[2], ratio)
    # GQA：kv_heads 减半，每 token 的 KV 字节减半 -> 同一 ceiling 翻倍
    mha = solve(context_tokens=2048, **{**args, "kv_heads": 8})
    gqa = solve(context_tokens=2048, **{**args, "kv_heads": 4})
    assert _close(gqa[3], 2.0 * mha[3]), (mha[3], gqa[3])
    # batch 再大也换不来 KV 占比下降之外的收益：权重被摊薄，weight_fraction 单调下降
    fractions = [solve(context_tokens=2048, **{**args, "batch": b})[6] for b in (1, 2, 8, 64)]
    assert all(a > c for a, c in zip(fractions, fractions[1:])), fractions
    assert _close(fractions[0], 13_200_000_000.0 / (13_200_000_000.0 + 268_435_456.0)), fractions[0]


def check_slo_min_batch_boundary(solve):
    args = (6_600_000_000, 32, 8, 128, 2048, 32, 312e12, 2e12)
    # target=0 -> 没有要求，batch=1 即满足
    assert solve(*args)[5] == 1
    assert solve(*args, target_tokens_per_s=0.0)[5] == 1
    low = solve(*args, target_tokens_per_s=500.0)[5]
    high = solve(*args, target_tokens_per_s=2000.0)[5]
    assert 1 <= low < high, (low, high)
    assert low < solve(*args, target_tokens_per_s=5000.0)[5], low
    # 目标超过上限：无论 batch 多大都达不到 -> -1
    ceiling = solve(*args)[3]
    assert solve(*args, target_tokens_per_s=1.5 * ceiling, max_batch=1000)[5] == -1
    # max_batch 太小：目标本身可达，但被 batch 上限卡住 -> -1
    need = solve(*args, target_tokens_per_s=2000.0)[5]
    assert solve(*args, target_tokens_per_s=2000.0, max_batch=need - 1)[5] == -1
    assert solve(*args, target_tokens_per_s=2000.0, max_batch=need)[5] == need
    # 达标判据本身：min_batch 满足、再小一个 batch 就不满足
    t = 2000.0
    reached = solve(6_600_000_000, 32, 8, 128, 2048, need, 312e12, 2e12)[0]
    missed = solve(6_600_000_000, 32, 8, 128, 2048, need - 1, 312e12, 2e12)[0]
    assert reached >= t > missed, (need, reached, missed)


def check_zero_context_and_degenerate(solve):
    # context_tokens=0：没有 KV，字节只剩权重，ceiling 取算力上限，权重占比恒 1
    got, want = _run(solve, (1000, 2, 2, 8, 0, 10, 1e9, 2e12))
    _assert_tuple(got, want)
    assert bool(got[1]) is False and _close(got[0], 500000.0), got[:2]
    assert _close(got[3], 1e9 / 2000.0) and int(got[4]) == 1 and _close(got[6], 1.0), got[3:5] + (got[6],)
    assert _close(got[2], 0.02), got[2]
    assert _close(got[0], got[3]), (got[0], got[3])              # 小模型纯算力封顶
    # 只涨 batch 不涨上下文：step_time 线性、tokens/s 恒等于算力上限
    a = solve(1000, 2, 2, 8, 0, 10, 1e9, 2e12)
    b = solve(1000, 2, 2, 8, 0, 20, 1e9, 2e12)
    assert _close(b[2], 2.0 * a[2]) and _close(b[0], a[0]), (a[2], b[2], a[0], b[0])
    # kv_bytes 减半（KV 存 fp8）：每 token 的 KV 字节减半 -> KV 型 ceiling 翻倍
    fp16kv = solve(6_600_000_000, 32, 8, 128, 2048, 1, 312e12, 2e12)
    fp8kv = solve(6_600_000_000, 32, 8, 128, 2048, 1, 312e12, 2e12, kv_bytes=1)
    assert _close(fp8kv[3], 2.0 * fp16kv[3]), (fp16kv[3], fp8kv[3])
    assert fp8kv[6] > fp16kv[6] > 0.98, (fp8kv[6], fp16kv[6])   # 权重仍占绝对主导
    # 最小结构（1 层 1 头 1 维、上下文 1）：2MB 权重远大于 4 字节 KV，拐点只有 11
    tiny = (1_000_000, 1, 1, 1, 1, 1, 1e12, 1e11)
    _assert_tuple(solve(*tiny), _oracle(*tiny))
    assert int(solve(*tiny)[4]) == 11 and bool(solve(*tiny)[1]) is True, solve(*tiny)
    assert _close(solve(*tiny)[6], 1.0), solve(*tiny)[6]
    # 上下文长到 KV 读彻底压不住：拐点判 -1，tokens/s 逼近但绝不越过 bw/K
    sat = solve(1_000_000, 32, 8, 128, 8192, 100000, 312e12, 2e12)
    assert int(sat[4]) == -1 and bool(sat[1]) is True, sat[4:5] + (sat[1],)
    assert _close(sat[3], 2e12 / 1_073_741_824.0), sat[3]
    assert _close(sat[0], sat[3], tol=0.01) and sat[0] < sat[3], (sat[0], sat[3])
    assert sat[6] < 1e-6, sat[6]                                # 权重被 10 万条序列摊没了


check_example.example = True
