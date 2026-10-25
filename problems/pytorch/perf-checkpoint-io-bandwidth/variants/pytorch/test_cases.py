import math


def _close(a, b, tol=1e-5):
    b = float(b)
    return abs(float(a) - b) <= tol * max(1.0, abs(b))


# 暴力模拟 staging 槽位轮转：每个检查点从排程时刻起占用一个槽位 copy+write 那么久，
# 若某次检查点要用的槽位还没空出来（写盘没结束），就判定「压不住」。
# 与参考解 (copy+write) > B*period 的闭合式结构完全不同，这里逐事件推演 60 次。
def _sim_writer_limited(copy_ms, write_ms, period_ms, buffers):
    slots = [0.0] * int(buffers)
    for c in range(60):
        scheduled = c * period_ms
        s = c % len(slots)
        if slots[s] > scheduled:
            return True
        slots[s] = max(slots[s], scheduled) + copy_ms + write_ms
    return False


# 独立重算（对拍口径）：state 用千元素块拆分再合并（不是直接相乘）、stall 用
# 「同步总时长 - 被异步藏掉的 write」、optimal 把 step_ms 除到根号里面去。
def _oracle(n_params, step_ms, checkpoint_interval, device_to_host_bw, storage_bw,
            async_save=True, staging_buffers=2, mtbf_steps=1e6, state_bytes_per_param=16):
    n = int(n_params)
    bps = int(state_bytes_per_param)
    state = (n // 1000) * (1000 * bps) + (n % 1000) * bps
    copy_ms = (state / float(device_to_host_bw)) * 1000.0
    write_ms = (state / float(storage_bw)) * 1000.0
    period_ms = float(checkpoint_interval) * float(step_ms)
    total_ms = copy_ms + write_ms
    stall_ms = total_ms - (write_ms if async_save else 0.0)
    overhead = stall_ms / period_ms
    if async_save:
        writer_limited = _sim_writer_limited(copy_ms, write_ms, period_ms, staging_buffers)
    else:                        # 同步：槽位占用不变，判据用 (copy+write)/B 与 period 比
        writer_limited = total_ms / float(staging_buffers) > period_ms
    arg = (2.0 * stall_ms / float(step_ms)) * float(mtbf_steps)
    optimal = int(math.sqrt(arg))
    if optimal < 1:
        optimal = 1
    return state, copy_ms, write_ms, stall_ms, period_ms, overhead, writer_limited, optimal


def _assert_tuple(got, want):
    assert len(got) == 8, got
    assert int(got[0]) == int(want[0]), (got[0], want[0])
    for k in (1, 2, 3, 4, 5):
        assert _close(got[k], want[k]), (got, want, k)
    assert bool(got[6]) == bool(want[6]), (got, want)
    assert int(got[7]) == int(want[7]), (got, want)


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    args = (1_000_000_000, 100.0, 200, 64e9, 10e9)
    _assert_tuple(solve(*args, True, 2, 1e6, 16), _oracle(*args, True, 2, 1e6, 16))
    _assert_tuple(solve(*args, False, 2, 1e6, 16), _oracle(*args, False, 2, 1e6, 16))
    got = solve(*args, True, 2, 1e6, 16)
    assert int(got[0]) == 16_000_000_000, got[0]
    assert _close(got[1], 250.0) and _close(got[2], 1600.0), (got[1], got[2])
    assert _close(got[3], 250.0) and _close(got[4], 20000.0), (got[3], got[4])
    assert _close(got[5], 0.0125) and bool(got[6]) is False, (got[5], got[6])
    assert int(got[7]) == 2236, got[7]
    got_sync = solve(*args, False, 2, 1e6, 16)
    assert _close(got_sync[3], 1850.0) and _close(got_sync[5], 0.0925), (got_sync[3], got_sync[5])
    assert int(got_sync[7]) == 6082, got_sync[7]


def check_matches_oracle(solve):
    cases = [
        (1_000_000_000, 100.0, 200, 64e9, 10e9, True, 2, 1e6, 16),
        (1_000_000_000, 100.0, 200, 64e9, 10e9, False, 2, 1e6, 16),
        (1_000_000_000, 100.0, 50, 64e9, 1e9, True, 2, 1e6, 16),      # 写盘压不住
        (1_000_000_000, 100.0, 50, 64e9, 1e9, False, 2, 1e6, 16),
        (500_000_000, 250.0, 1, 32e9, 2e9, True, 1, 1e5, 16),
        (1_000_000, 10.0, 1000, 16e9, 5e8, False, 4, 0.5, 16),        # 触发钳位
        (7_000_000_000, 137.0, 300, 50e9, 4e9, True, 2, 4e5, 8),
        (1_000_000_000, 100.0, 200, 1e30, 1e30, True, 2, 1e6, 16),    # 带宽无穷
        (123_456_789, 88.0, 77, 21e9, 3.3e9, True, 3, 2.5e5, 12),
        (2_000_000_000, 200.0, 7, 32e9, 16e9, True, 2, 1e6, 16),
        (2_000_000_000, 200.0, 7, 32e9, 16e9, False, 2, 1e6, 16),
    ]
    for c in cases:
        _assert_tuple(solve(*c), _oracle(*c))


def check_async_only_pays_the_blocking_copy(solve):
    # 异步只付 D2H 拷贝；同步把落盘也串进关键路径
    args = (2_000_000_000, 200.0, 100, 40e9, 5e9)
    _, copy_ms, write_ms, stall_async, _, ovh_async, _, _ = solve(*args, True, 2, 1e6, 16)
    _, _, _, stall_sync, _, ovh_sync, _, _ = solve(*args, False, 2, 1e6, 16)
    assert _close(stall_async, copy_ms), (stall_async, copy_ms)
    assert _close(stall_sync, copy_ms + write_ms), (stall_sync, copy_ms + write_ms)
    assert _close(stall_sync - stall_async, write_ms), (stall_sync, stall_async)
    assert ovh_sync > ovh_async, (ovh_sync, ovh_async)
    assert _close(ovh_async, copy_ms / (100 * 200.0)), ovh_async


def check_interval_one_is_worst_case(solve):
    n, step, d2h, stw = 1_000_000_000, 100.0, 64e9, 10e9
    # 每步都存：overhead = (copy+write)/step，且远超 1 -> 完全不可行
    _, copy_ms, write_ms, stall_ms, period_ms, overhead, limited, _ = solve(
        n, step, 1, d2h, stw, False, 2, 1e6, 16)
    assert _close(period_ms, 100.0), period_ms
    assert _close(stall_ms, copy_ms + write_ms) and _close(overhead, stall_ms / 100.0), (stall_ms, overhead)
    assert overhead > 1.0 and bool(limited) is True, (overhead, limited)
    # 周期加倍：period 加倍、overhead 减半、stall 不变
    _, _, _, stall2, period2, overhead2, _, _ = solve(n, step, 2, d2h, stw, False, 2, 1e6, 16)
    assert _close(period2, 200.0) and _close(stall2, stall_ms), (period2, stall2)
    assert _close(overhead2 * 2.0, overhead), (overhead2, overhead)


def check_writer_limited_threshold(solve):
    # copy=500ms, write=1000ms：一次检查点占用槽位 1500ms，B 个槽位容忍落后 B 个周期
    n, step, d2h, stw = 1_000_000_000, 100.0, 32e9, 16e9
    for interval, buffers, expect in [(8, 1, True),       # period=800  -> 1500 > 800
                                      (8, 2, False),      # 1500 <= 1600
                                      (1, 16, False),     # 1500 <= 1600
                                      (1, 15, False),     # 1500 > 1500 为假（严格大于）
                                      (15, 1, False),     # period=1500 -> 持平不算压不住
                                      (14, 1, True)]:     # 1500 > 1400
        limited = solve(n, step, interval, d2h, stw, True, buffers, 1e6, 16)[6]
        assert bool(limited) is expect, (interval, buffers, limited)
    # 槽位占用与是否重叠无关：同一配置下 async/sync 的判据一致
    a = solve(n, step, 7, d2h, stw, True, 2, 1e6, 16)[6]
    b = solve(n, step, 7, d2h, stw, False, 2, 1e6, 16)[6]
    assert bool(a) is True and bool(b) is True, (a, b)


def check_optimal_interval_clamps_to_one(solve):
    # mtbf 很小 / 检查点很便宜 -> 开方后不足 1，按「至少每步一存」钳到 1
    got = solve(1_000_000, 100.0, 1000, 64e9, 1e30, True, 2, 0.5, 16)
    assert int(got[7]) == 1, got
    assert _close(got[3], 1000.0 * 16e6 / 64e9), got[3]
    # 同步模式下存储越慢、阻塞越久、最优周期越长（Daly 的 sqrt 规律）
    cheap = solve(1_000_000_000, 100.0, 200, 64e9, 1e12, False, 2, 1e6, 16)[7]
    pricey = solve(1_000_000_000, 100.0, 200, 64e9, 1e9, False, 2, 1e6, 16)[7]
    assert cheap < pricey, (cheap, pricey)
    # mtbf 翻倍 -> 周期约乘 sqrt(2)
    short = solve(1_000_000_000, 100.0, 200, 64e9, 1e9, False, 2, 1e6, 16)[7]
    long_ = solve(1_000_000_000, 100.0, 200, 64e9, 1e9, False, 2, 2e6, 16)[7]
    assert long_ > short, (short, long_)
    assert abs(long_ / short - math.sqrt(2.0)) < 0.02, (short, long_)


def check_infinite_bandwidth_degenerate(solve):
    # 带宽无穷大：没有拷贝也没有落盘，开销为 0、压不住为假、周期钳到 1
    got = solve(7_000_000_000, 50.0, 100, 1e30, 1e30, False, 2, 1e6, 16)
    assert _close(got[1], 0.0) and _close(got[2], 0.0) and _close(got[3], 0.0), got[:4]
    assert _close(got[5], 0.0) and bool(got[6]) is False and int(got[7]) == 1, got[5:]


def check_state_bytes_follow_training_state(solve):
    # 字节口径沿用 training-state-memory：默认 16 字节/参数（fp16 权重+梯度、fp32 master+m+v）
    assert int(solve(1_000_000_000, 100.0, 10, 64e9, 10e9)[0]) == 16_000_000_000
    assert int(solve(1_000_000_000, 100.0, 10, 64e9, 10e9, state_bytes_per_param=8)[0]) == 8_000_000_000
    # 状态字节减半 -> copy/write/stall/overhead 全部减半
    a = solve(1_000_000_000, 100.0, 10, 64e9, 10e9, False, 2, 1e6, 16)
    b = solve(1_000_000_000, 100.0, 10, 64e9, 10e9, False, 2, 1e6, 8)
    for k in (1, 2, 3, 5):
        assert _close(a[k], 2.0 * b[k]), (k, a[k], b[k])


check_example.example = True
