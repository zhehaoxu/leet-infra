def _close(a, b, tol=1e-6):
    b = float(b)
    return abs(float(a) - b) <= tol * max(1.0, abs(b))


# 独立参考：对角填充前向扫描，数忙碌格算利用率（不直接写闭式公式）
def _ref(p, m):
    if p < 1 or m < 1:
        raise ValueError
    makespan = m + p - 1
    busy = [[False] * makespan for _ in range(p)]
    for i in range(m):                 # 第 i 个 micro-batch
        for j in range(p):             # 第 j 段流水线
            busy[j][i + j] = True      # 前向落在对角时间槽
    total_busy = sum(sum(row) for row in busy)
    util = total_busy / (p * makespan)
    bubble = 1.0 - util
    return bubble, util, float(makespan)


# 样例用例
def check_example(solve):
    bubble, util, makespan = solve(4, 8)
    assert _close(makespan, 11.0), makespan
    assert _close(util, 8.0 / 11.0), util
    assert _close(bubble, 3.0 / 11.0), bubble


def check_bubble_plus_util_one(solve):
    for p, m in [(2, 2), (4, 8), (8, 4), (3, 32), (16, 3)]:
        bubble, util, _ = solve(p, m)
        assert _close(bubble + util, 1.0), (p, m, bubble, util)


def check_single_stage(solve):
    # p=1：无流水线，全利用
    bubble, util, makespan = solve(1, 5)
    assert bubble == 0.0 and util == 1.0 and _close(makespan, 5.0), (bubble, util, makespan)


def check_single_microbatch(solve):
    # m=1：完全串行，util=1/p, bubble=(p-1)/p
    for p in (2, 4, 8):
        bubble, util, makespan = solve(p, 1)
        assert _close(makespan, float(p)), makespan
        assert _close(util, 1.0 / p), util
        assert _close(bubble, (p - 1.0) / p), bubble


def check_more_microbatches_less_bubble(solve):
    # 固定 p，m 越大气泡越小（单调）
    p = 6
    prev = 2.0
    for m in (1, 2, 4, 8, 16, 64):
        bubble, _, _ = solve(p, m)
        assert bubble <= prev + 1e-12, (m, bubble, prev)
        prev = bubble


def check_matches_ref(solve):
    for p, m in [(4, 8), (3, 5), (8, 16), (2, 2), (5, 1), (1, 7), (12, 24)]:
        got = solve(p, m)
        ref = _ref(p, m)
        assert len(got) == 3
        for g, r in zip(got, ref):
            assert _close(g, r), (p, m, got, ref)


check_example.example = True
