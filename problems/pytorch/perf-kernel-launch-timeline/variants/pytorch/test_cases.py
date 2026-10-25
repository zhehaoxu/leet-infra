def _close(a, b, tol=1e-5):
    b = float(b)
    return abs(float(a) - b) <= tol * max(1.0, abs(b))


# 独立重算（对拍口径）：eager 完成时刻用「单机 + 释放时间」的经典 max-plus 闭式
#   e_i = max_{0<=j<=i} [ (j+1)*L + sum_{k=j..i} d_k ]
# 即枚举每个起点 j「从 j 号 kernel 起跑」的整段耗时取最大（三重循环暴力），
# 与参考解的正向逐步递推结构完全不同；空转个数再用 e_{i-1} 与就绪时刻比较。
def _oracle(durations_us, launch_overhead_us, use_graph):
    d = [float(x) for x in durations_us]
    L = float(launch_overhead_us)
    n = len(d)
    ends = []
    for i in range(n):
        best = 0.0
        for j in range(i + 1):
            traffic = 0.0
            for k in range(j, i + 1):
                traffic += d[k]
            cand = (j + 1) * L + traffic
            if cand > best:
                best = cand
        ends.append(best)
    eager = ends[-1]
    graph = L + sum(d)
    launch_bound = 0
    for i in range(n):
        prev_end = ends[i - 1] if i > 0 else 0.0
        if (i + 1) * L > prev_end:
            launch_bound += 1
    wall = graph if use_graph else eager
    work = sum(d)
    busy = work / wall if wall > 0.0 else 1.0
    speedup = eager / graph if graph > 0.0 else 1.0
    return eager, graph, wall, busy, launch_bound, speedup


def _assert_tuple(got, want):
    assert len(got) == 6, got
    for k in (0, 1, 2, 3, 5):
        assert _close(got[k], want[k]), (got, want, k)
    assert int(got[4]) == int(want[4]), (got, want)


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    _assert_tuple(solve([30.0, 5.0, 5.0, 20.0], 20.0, False),
                  (100.0, 80.0, 100.0, 0.6, 3, 1.25))
    _assert_tuple(solve([30.0, 5.0, 5.0, 20.0], 20.0, True),
                  (100.0, 80.0, 80.0, 0.75, 3, 1.25))


def check_matches_oracle(solve):
    cases = [
        ([30.0, 5.0, 5.0, 20.0], 20.0, False),
        ([30.0, 5.0, 5.0, 20.0], 20.0, True),
        ([30.0, 5.0, 5.0, 20.0], 10.0, False),
        ([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 4.0, False),
        ([100.0, 2.0, 3.0], 10.0, True),
        ([0.5, 0.5, 0.5], 0.5, False),
        ([8.0, 8.0], 8.0, True),
        ([2.0, 40.0, 1.0, 1.0, 30.0, 3.0], 6.0, False),
    ]
    for durations, L, g in cases:
        _assert_tuple(solve(durations, L, g), _oracle(durations, L, g))


def check_single_kernel_no_gain(solve):
    # 只有一个 kernel：eager 与 graph 都要等一次发射，CUDA Graph 毫无收益
    eager, graph, wall, busy, cnt, speed = solve([25.0], 7.0, True)
    assert _close(eager, 32.0) and _close(graph, 32.0) and _close(wall, 32.0), (eager, graph, wall)
    assert _close(speed, 1.0), speed
    assert _close(busy, 25.0 / 32.0), busy
    assert int(cnt) == 1, cnt


def check_zero_overhead_is_busy(solve):
    # 发射开销为 0：没有任何空转，busy=1，graph 无收益
    durations = [3.0, 7.0, 1.5, 2.0]
    eager, graph, _, busy, cnt, speed = solve(durations, 0.0, False)
    total = sum(durations)
    assert _close(eager, total) and _close(graph, total), (eager, graph)
    assert _close(busy, 1.0) and int(cnt) == 0 and _close(speed, 1.0), (busy, cnt, speed)


def check_pure_launch_bound(solve):
    # kernel 时长全为 0：墙钟被发射次数完全决定，打包成图的收益正好是 n 倍
    eager, graph, wall, busy, cnt, speed = solve([0.0, 0.0, 0.0, 0.0], 7.0, False)
    assert _close(eager, 28.0) and _close(graph, 7.0), (eager, graph)
    assert _close(wall, 28.0) and int(cnt) == 4, (wall, cnt)
    assert _close(busy, 0.0) and _close(speed, 4.0), (busy, speed)
    _, _, walled, busy_g, _, _ = solve([0.0, 0.0, 0.0, 0.0], 7.0, True)
    assert _close(walled, 7.0) and _close(busy_g, 0.0), (walled, busy_g)


def check_long_kernels_only_first_gap(solve):
    # kernel 都足够长：只有首发射造成一次空转，eager 与 graph 完全相同
    durations = [100.0, 200.0, 50.0, 400.0]
    eager, graph, _, busy, cnt, speed = solve(durations, 10.0, False)
    total = sum(durations)
    assert _close(eager, 10.0 + total) and _close(graph, 10.0 + total), (eager, graph)
    assert int(cnt) == 1 and _close(speed, 1.0), (cnt, speed)
    assert _close(busy, total / (10.0 + total)), busy


def check_idle_gap_identity(solve):
    # busy 口径不变量：busy_ratio * wall == 有效计算时长，且恒 <= 1
    for durations, L, g in [([30.0, 5.0, 5.0, 20.0], 10.0, False),
                            ([30.0, 5.0, 5.0, 20.0], 10.0, True),
                            ([1.0, 1.0, 1.0], 4.0, False),
                            ([0.0, 3.0, 0.0], 5.0, True)]:
        _, _, wall, busy, _, _ = solve(durations, L, g)
        work = sum(durations)
        if wall > 0.0:
            assert _close(busy * wall, work), (durations, L, g, busy, wall)
            assert busy <= 1.0 + 1e-12, (durations, L, g, busy)


def check_graph_never_slower(solve):
    for durations, L in [([30.0, 5.0, 5.0, 20.0], 10.0), ([1.0] * 8, 5.0),
                         ([0.0, 0.0], 3.0), ([50.0, 50.0], 1.0)]:
        eager, graph, _, _, _, speed = solve(durations, L, False)
        assert eager >= graph - 1e-12, (durations, L, eager, graph)
        assert speed >= 1.0 - 1e-12, (durations, L, speed)


def check_zero_everything_guards(solve):
    # 零时长 kernel + 零发射开销：两处除法分母都是 0，按约定记 1.0
    eager, graph, wall, busy, cnt, speed = solve([0.0, 0.0], 0.0, False)
    assert _close(eager, 0.0) and _close(graph, 0.0) and _close(wall, 0.0), (eager, graph, wall)
    assert _close(busy, 1.0) and int(cnt) == 0 and _close(speed, 1.0), (busy, cnt, speed)
    # 只有首 kernel 空转、后续背靠背
    eager, graph, _, _, cnt, _ = solve([4.0, 4.0, 4.0], 2.0, False)
    assert int(cnt) == 1 and _close(eager, 14.0), (cnt, eager)


check_example.example = True
