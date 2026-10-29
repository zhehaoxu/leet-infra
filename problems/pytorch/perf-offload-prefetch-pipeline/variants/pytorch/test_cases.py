# 独立重算（对拍口径）：不看公式，直接按「单位时间 tick」逐拍推进状态机——
# 引擎每拍搬 bw 字节直到该层剩余字节清零，缓冲槽用「占用标记」从预取开始占到该层计算结束，
# 计算与传输在同一拍内并行。空等时间与 makespan 都由时间线自然涌现。

def _oracle(compute, transfer_bytes, bw, buffers):
    layers = len(compute)
    bw = int(bw)
    buffers = int(buffers)

    held = [False] * buffers
    prefetched = [False] * layers
    transfer_start = [0] * layers
    transfer_end = [0] * layers
    compute_start = [0] * layers

    active = None            # [layer, remaining_bytes]
    nxt_prefetch = 0
    cur = 0
    left = None
    done = 0
    t = 0

    while done < layers:
        # 1) 引擎空闲且目标槽未被占用时发起预取；0 字节的层同一拍内立刻完成并可继续链式发起
        while active is None and nxt_prefetch < layers and not held[nxt_prefetch % buffers]:
            j = nxt_prefetch
            nxt_prefetch += 1
            transfer_start[j] = t
            held[j % buffers] = True
            nbytes = int(transfer_bytes[j])
            if nbytes <= 0:
                transfer_end[j] = t
                prefetched[j] = True
                continue
            active = [j, nbytes]

        # 2) 本层已就位（预取在本拍之前完成）且上一步计算已结束，则本拍开始计算
        if left is None and cur < layers and prefetched[cur]:
            compute_start[cur] = t
            left = int(compute[cur])

        # 3) 传输推进一拍
        if active is not None:
            active[1] -= bw
            if active[1] <= 0:
                transfer_end[active[0]] = t + 1
                prefetched[active[0]] = True
                active = None

        # 4) 计算推进一拍；结束的那一拍释放缓冲槽
        if left is not None:
            left -= 1
            if left == 0:
                held[cur % buffers] = False
                done += 1
                cur += 1
                left = None

        t += 1

    makespan = t
    stall = makespan - sum(int(x) for x in compute)
    return (int(makespan), int(stall), [int(x) for x in compute_start],
            [int(x) for x in transfer_start], [int(x) for x in transfer_end])


def _assert(got, want):
    assert len(got) == 5, got
    for k in (0, 1):
        assert int(got[k]) == int(want[k]), (k, got, want)
    for k in (2, 3, 4):
        assert [int(x) for x in got[k]] == [int(x) for x in want[k]], (k, got[k], want[k])


# 样例：bw=50，四层，transfer_time = [2,4,1,3]，buffers=2
#   j0: s=0  e=2  c=2  finish=7            槽0 从 0 占到 7
#   j1: s=2  e=6  c=max(7,6)=7 finish=10   槽1 从 2 占到 10
#   j2: 槽0 在 finish[0]=7 释放 -> s=7 e=8 c=10 finish=14
#   j3: 槽1 在 finish[1]=10 释放 -> s=10 e=13 c=14 finish=16
#   makespan=16 = Σcompute 14 + 空等 2（只有首层干等了第一次预取）
def check_example(solve):
    got = solve([5, 3, 4, 2], [100, 200, 50, 150], 50, 2)
    _assert(got, (16, 2, [2, 7, 10, 14], [0, 2, 7, 10], [2, 6, 8, 13]))


check_example.example = True


def check_matches_tick_oracle(solve):
    import random

    rng = random.Random(77)
    for _ in range(30):
        layers = rng.randrange(1, 7)
        compute = [rng.randrange(1, 9) for _ in range(layers)]
        transfer = [rng.choice([0, rng.randrange(1, 300)]) for _ in range(layers)]
        bw = rng.randrange(1, 61)
        buffers = rng.randrange(1, 4)
        _assert(solve(compute, transfer, bw, buffers), _oracle(compute, transfer, bw, buffers))


def check_single_buffer_is_serial(solve):
    # 单槽无法重叠：每层都要等上一层算完才能开始预取，空等 = 全部传输时间
    compute = [4, 2, 5]
    transfer = [30, 60, 10]
    bw = 10
    got = solve(compute, transfer, bw, 1)
    _assert(got, (21, 10, [3, 13, 16], [0, 7, 15], [3, 13, 16]))


def check_zero_transfer(solve):
    # 0 字节层的「预取」不占引擎，但单槽仍要等上一层算完才能发起
    got = solve([3, 5, 2], [0, 0, 0], 64, 1)
    _assert(got, (10, 0, [0, 3, 8], [0, 3, 8], [0, 3, 8]))


def check_accounting_identity(solve):
    import random

    rng = random.Random(13)
    for _ in range(20):
        layers = rng.randrange(1, 8)
        compute = [rng.randrange(1, 10) for _ in range(layers)]
        transfer = [rng.randrange(0, 400) for _ in range(layers)]
        bw = rng.randrange(1, 50)
        buffers = rng.randrange(1, 4)
        makespan, stall, cstart, tstart, tend = solve(compute, transfer, bw, buffers)
        assert int(makespan) == sum(compute) + int(stall), (makespan, stall, compute)
        for j in range(layers):
            assert int(tend[j]) - int(tstart[j]) == (-(-transfer[j] // bw) if transfer[j] > 0 else 0), j
            assert int(cstart[j]) >= int(tend[j]), (j, cstart, tend)
        finish = 0
        for j in range(layers):
            assert int(cstart[j]) >= finish, (j, cstart)
            finish = int(cstart[j]) + compute[j]


def check_slot_occupancy(solve):
    import random

    rng = random.Random(21)
    for _ in range(15):
        layers = rng.randrange(2, 8)
        compute = [rng.randrange(1, 8) for _ in range(layers)]
        transfer = [rng.randrange(1, 200) for _ in range(layers)]
        bw = rng.randrange(1, 30)
        buffers = rng.randrange(1, 4)
        _, _, cstart, tstart, _ = solve(compute, transfer, bw, buffers)
        for j in range(buffers, layers):
            slot_freed = int(cstart[j - buffers]) + compute[j - buffers]
            assert int(tstart[j]) >= slot_freed, (j, buffers, tstart[j], slot_freed)


def check_more_buffers_never_worse(solve):
    import random

    rng = random.Random(31)
    for _ in range(10):
        layers = rng.randrange(2, 7)
        compute = [rng.randrange(1, 8) for _ in range(layers)]
        transfer = [rng.randrange(1, 200) for _ in range(layers)]
        bw = rng.randrange(1, 25)
        prev = None
        for buffers in (1, 2, 3):
            makespan = int(solve(compute, transfer, bw, buffers)[0])
            if prev is not None:
                assert makespan <= prev, (compute, transfer, bw, buffers, makespan, prev)
            prev = makespan


def check_large_compute_hides_transfer(solve):
    # 计算远长于传输：除首层外计算不等传输；但第 3 层仍要等槽 0 在 finish[0]=11 释放
    got = solve([10, 10, 10], [40, 40, 40], 40, 2)
    _assert(got, (31, 1, [1, 11, 21], [0, 1, 11], [1, 2, 12]))
    assert solve([], [], 8, 2) == (0, 0, [], [], [])
