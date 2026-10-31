from itertools import combinations


def _py(value):
    """oracle：穷举全部单调切分（把 T_mel 帧分给 T_text 个 token 的单调组合）取 max。"""
    T_mel = len(value)
    T_text = len(value[0])
    best_log = None
    best_assign = None
    # 切分点组合：选 T_text-1 个切换帧（帧 i 后切到下一 token）
    for cuts in combinations(range(T_mel - 1), T_text - 1):
        bounds = [-1] + list(cuts) + [T_mel - 1]
        log = 0.0
        assign = [[0] * T_text for _ in range(T_mel)]
        durs = []
        for s in range(T_text):
            lo, hi = bounds[s] + 1, bounds[s + 1]
            durs.append(hi - lo + 1)
            for t in range(lo, hi + 1):
                log += value[t][s]
                assign[t][s] = 1
        if best_log is None or log > best_log:
            best_log = log
            best_assign = (assign, durs)
    return {"log_prob": best_log, "attn": best_assign[0],
            "durations": best_assign[1]}


def _compare(solve, value):
    led = solve(value)
    ref = _py(value)
    assert led == ref, f"solve vs oracle:\n{led}\nvs\n{ref}"
    return led


def check_anchor_basic(solve):
    # 手算：2 帧 2 token，value [[1, 0], [0, 2]]
    # 路径：token0 两帧 1+0=1；token0 帧0 + token1 帧1 = 3；token1 两帧不可（单调可：帧0 归 s1? s0 必须有帧——
    # 单调路径必须终点 (1,1)，帧0 可归 s0 或 s1？dp[0][1] 不可达（首行仅 (0,0)）→ 帧0 必归 s0
    led = _compare(solve, [[1.0, 0.0], [0.0, 2.0]])
    assert led["log_prob"] == 3.0
    assert led["attn"] == [[1, 0], [0, 1]]
    assert led["durations"] == [1, 1]


def check_anchor_repeat_token(solve):
    # 3 帧 1 token：全归 s0
    led = _compare(solve, [[0.5], [1.5], [2.5]])
    assert led["log_prob"] == 4.5
    assert led["durations"] == [3]
    assert led["attn"] == [[1], [1], [1]]


def check_greedy_not_optimal(solve):
    # 贪心（逐帧取局部最优 token）≠ 全局最优的构造：
    # token0 帧单值高但抢走帧后 token1 损失更大
    value = [[5.0, 0.0], [5.0, 0.0], [0.0, 100.0]]
    led = _compare(solve, value)
    # 最优：t0,t1 归 s0 (10)，t2 归 s1 (100) → 110
    assert led["log_prob"] == 110.0
    assert led["durations"] == [2, 1]


def check_tie_any_path_ok(solve):
    # 完全对称：log_prob 唯一，attn 任一最优解（oracle 穷举取第一组 max，与 DP 回溯一致即可）
    led = _compare(solve, [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    assert led["log_prob"] == 3.0
    assert sum(led["durations"]) == 3


def check_long_vs_short(solve):
    # T_mel == T_text：路径唯一（每帧必切换）
    value = [[1.0, 3.0, 0.0], [2.0, 0.0, 4.0], [0.0, 0.0, 5.0]]
    led = _compare(solve, value)
    assert led["log_prob"] == 6.0  # 对角 1 + 0 + 5
    assert led["durations"] == [1, 1, 1]


def check_invalid_inputs(solve):
    def ve(*args):
        try:
            solve(*args)
        except ValueError:
            return
        raise AssertionError("expected ValueError")
    ve([])
    ve([[]])
    ve([[1.0, 2.0], [1.0]])            # 行长不齐
    ve([[1.0, "x"]])
    ve([[1.0, float("nan")]])
    ve([[1.0, 2.0]])                    # T_mel < T_text：token 无帧
    ve("nope")


def check_oracle_random(solve):
    import random
    rng = random.Random(20260672)
    for _ in range(200):
        T_text = rng.randint(1, 4)
        T_mel = rng.randint(T_text, 8)
        value = [[round(rng.uniform(-3, 3), 3) for _ in range(T_text)]
                 for _ in range(T_mel)]
        led = solve(value)
        ref = _py(value)
        assert abs(led["log_prob"] - ref["log_prob"]) < 1e-9, \
            f"mismatch: {led['log_prob']} vs {ref['log_prob']}"
        assert sum(led["durations"]) == T_mel
        assert all(d >= 1 for d in led["durations"])
        # attn 与 durations 一致
        col = [sum(row[s] for row in led["attn"]) for s in range(T_text)]
        assert col == led["durations"]
        # 重算 log：attn 加权求和 == log_prob
        total = sum(value[t][s] for t in range(T_mel) for s in range(T_text)
                    if led["attn"][t][s] == 1)
        assert abs(total - led["log_prob"]) < 1e-9
