import math

import torch


def _close(a, b, tol=1e-5):
    if math.isinf(float(b)):
        return math.isinf(float(a)) and (a > 0) == (b > 0)
    b = float(b)
    return abs(float(a) - b) <= tol * max(1.0, abs(b))


# 独立重算（对拍口径）：Amdahl 不用 total-p+p/k，而是显式造出「改造后的相位表」再整体求和；
# compute 走 total - phase_ms[0] 这条路（参考解用的是 1+2+3 三项相加）。
def _oracle(phase_ms, target_phase, speedup, worker_ms, num_workers, prefetch):
    p = [float(x) for x in phase_ms]
    total = 0.0
    for x in p:
        total += x
    fracs = torch.tensor([x / total for x in p], dtype=torch.float64)

    t = int(target_phase)
    modified = []
    for i, x in enumerate(p):
        modified.append(x / float(speedup) if i == t else x)
    new_step = 0.0
    for x in modified:
        new_step += x
    rest = 0.0
    for i, x in enumerate(p):
        if i != t:
            rest += x
    amdahl = total / new_step if new_step > 0.0 else float("inf")
    ceiling = total / rest if rest > 0.0 else float("inf")

    supply = float(worker_ms) / float(num_workers)
    compute = total - p[0]
    deficit = supply - compute
    if int(prefetch) == 0:
        wait = supply
    elif deficit > 0.0:
        wait = deficit
    else:
        wait = 0.0
    effective = compute + wait
    hidden = (wait == 0.0)
    return fracs, total, new_step, amdahl, ceiling, effective, hidden


def _assert_tuple(got, want):
    assert len(got) == 7, got
    assert torch.allclose(got[0], want[0], atol=1e-5, rtol=1e-5), (got[0], want[0])
    for k in (1, 2, 3, 4, 5):
        assert _close(got[k], want[k]), (got, want, k)
    assert bool(got[6]) == bool(want[6]), (got, want)


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    got = solve([20.0, 100.0, 200.0, 40.0], 0, 4.0, 80.0, 4, 2)
    want = _oracle([20.0, 100.0, 200.0, 40.0], 0, 4.0, 80.0, 4, 2)
    _assert_tuple(got, want)
    _assert_tuple(got, (torch.tensor([20 / 360, 100 / 360, 200 / 360, 40 / 360], dtype=torch.float64),
                        360.0, 345.0, 360.0 / 345.0, 360.0 / 340.0, 340.0, True))
    # prefetch=0：取数完全串进关键路径，掩盖不了
    got2 = solve([150.0, 60.0, 120.0, 20.0], 0, 8.0, 90.0, 3, 0)
    _assert_tuple(got2, (torch.tensor([150 / 350, 60 / 350, 120 / 350, 20 / 350], dtype=torch.float64),
                         350.0, 218.75, 1.6, 1.75, 230.0, False))


def check_matches_oracle(solve):
    cases = [
        ([20.0, 100.0, 200.0, 40.0], 0, 4.0, 80.0, 4, 2),
        ([20.0, 100.0, 200.0, 40.0], 2, 3.0, 80.0, 4, 2),
        ([150.0, 60.0, 120.0, 20.0], 0, 8.0, 90.0, 3, 0),
        ([0.0, 6.25, 2.5, 1.25], 1, 2.0, 20.0, 2, 1),
        ([10.0, 30.0, 60.0, 5.0], 3, 10.0, 4.0, 4, 8),
        ([5.0, 5.0, 5.0, 5.0], 1, 1.0, 12.0, 1, 0),
        ([7.5, 12.0, 48.0, 2.5], 2, 1.5, 30.0, 6, 3),
    ]
    for c in cases:
        _assert_tuple(solve(*c), _oracle(*c))


def check_k_one_is_identity(solve):
    # 不提速：step 时长不变、加速比恰为 1
    for phases, t in [([20.0, 100.0, 200.0, 40.0], 0), ([20.0, 100.0, 200.0, 40.0], 2),
                      ([3.0, 7.0, 11.0, 2.0], 3)]:
        total = sum(phases)
        fr, tot, new_step, amdahl, ceil, eff, hidden = solve(phases, t, 1.0, 50.0, 2, 1)
        assert _close(tot, total) and _close(new_step, total), (tot, new_step)
        assert _close(amdahl, 1.0), amdahl
        assert ceil >= amdahl, (ceil, amdahl)


def check_amdahl_ceiling_is_the_limit(solve):
    # k 趋于无穷时整机加速比逼近上限 total/(total-p)
    phases = [20.0, 100.0, 200.0, 40.0]
    for t in range(4):
        _, _, _, amdahl_big, ceiling, _, _ = solve(phases, t, 1e9, 1.0, 1, 1)
        assert _close(amdahl_big, ceiling, 1e-4), (t, amdahl_big, ceiling)
    # 上限随被提速相位占比单调：占比最大（backward）的上限最高
    ceilings = [solve(phases, t, 1e9, 1.0, 1, 1)[4] for t in range(4)]
    assert ceilings[2] == max(ceilings), ceilings
    assert min(ceilings) > 1.0, ceilings


def check_only_phase_gives_infinite_ceiling(solve):
    # 其余相位全为 0：上限是无穷，整机加速比恰好等于 k
    fr, tot, new_step, amdahl, ceiling, eff, hidden = solve([10.0, 0.0, 0.0, 0.0], 0, 5.0, 0.0, 1, 1)
    assert math.isinf(ceiling) and ceiling > 0, ceiling
    assert _close(tot, 10.0) and _close(new_step, 2.0), (tot, new_step)
    assert _close(amdahl, 5.0), amdahl
    assert _close(eff, 0.0) and bool(hidden) is True, eff


def check_prefetch_zero_has_no_overlap(solve):
    # 同步取数：effective = compute + supply，永远掩盖不了
    fr, tot, ns, am, ce, eff, hidden = solve([40.0, 60.0, 120.0, 20.0], 0, 2.0, 60.0, 2, 0)
    supply = 60.0 / 2
    compute = 60.0 + 120.0 + 20.0
    assert _close(eff, compute + supply), (eff, compute + supply)
    assert bool(hidden) is False, hidden
    # 同样配置开一个预取位：供得上就完全不付取数代价
    _, _, _, _, _, eff1, hidden1 = solve([40.0, 60.0, 120.0, 20.0], 0, 2.0, 60.0, 2, 1)
    assert _close(eff1, compute) and bool(hidden1) is True, (eff1, hidden1)


def check_more_workers_never_hurt(solve):
    phases = [80.0, 6.25, 2.5, 1.25]
    compute = 6.25 + 2.5 + 1.25
    prev = None
    for w in (1, 2, 3, 4, 8, 16, 64):
        eff, hidden = solve(phases, 0, 2.0, 80.0, w, 4)[5], solve(phases, 0, 2.0, 80.0, w, 4)[6]
        if prev is not None:
            assert eff <= prev + 1e-9, (w, eff, prev)
        prev = eff
    # worker 多到 supply <= compute 后被完全掩盖，稳态等于纯计算
    assert _close(prev, compute) and bool(solve(phases, 0, 2.0, 80.0, 64, 4)[6]) is True, prev


def check_fractions_sum_to_one(solve):
    for phases in ([20.0, 100.0, 200.0, 40.0], [1.0, 1.0, 1.0, 1.0], [0.0, 5.0, 0.0, 3.0]):
        fr = solve(phases, 0, 2.0, 10.0, 2, 1)[0]
        assert fr.shape == (4,), fr
        assert abs(float(fr.sum()) - 1.0) <= 1e-9, fr
        tot = float(fr[1] + fr[2] + fr[3])
        assert abs(tot + float(fr[0]) - 1.0) <= 1e-9, fr


def check_loader_pipeline_simulation(solve):
    # 暴力模拟 S 步取数-计算流水，稳态摊销应与闭合式一致
    steps = 4000
    for worker_ms, nw, prefetch, phases in [(60.0, 2, 4, [50.0, 6.25, 2.5, 1.25]),
                                            (10.0, 2, 4, [50.0, 6.25, 2.5, 1.25]),
                                            (0.0, 8, 2, [50.0, 6.25, 2.5, 1.25])]:
        supply = worker_ms / nw
        compute = sum(phases[1:])
        end = 0.0
        for j in range(steps):
            ready = (j + 1) * supply
            start = ready if ready > end else end
            end = start + compute
        sim = end / steps
        eff = solve(phases, 0, 2.0, worker_ms, nw, prefetch)[5]
        assert abs(sim - eff) <= 1e-3 * max(1.0, eff), (worker_ms, nw, sim, eff)


check_example.example = True
