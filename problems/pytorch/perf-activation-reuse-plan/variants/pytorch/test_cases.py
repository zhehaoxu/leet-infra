# 独立重算（对拍口径）：偏移不用「候选集合」，而是从 0 起**跳跃式右移**——
# 一旦当前偏移撞上某个已放置块，就直接跳到该块末端（多块冲突取最远末端）再试，
# 首次不再冲突的位置就是最小可行偏移；下界改用事件扫描（end 先于 start 结算）求和。

def _oracle(intervals, align):
    align = int(align)

    def round_up(n):
        return -(-int(n) // align) * align

    n = len(intervals)
    starts = [int(iv[0]) for iv in intervals]
    ends = [int(iv[1]) for iv in intervals]
    sizes = [round_up(iv[2]) for iv in intervals]
    order = sorted(range(n), key=lambda i: (-sizes[i], starts[i], ends[i], i))

    placed = []
    offsets = [0] * n
    for i in order:
        off = 0
        while True:
            jump = None
            for ps, pe, poff, psize in placed:
                if max(starts[i], ps) >= min(ends[i], pe):
                    continue
                if max(off, poff) < min(off + sizes[i], poff + psize):
                    end = poff + psize
                    jump = end if jump is None else max(jump, end)
            if jump is None:
                break
            off = jump
        offsets[i] = off
        placed.append((starts[i], ends[i], off, sizes[i]))

    peak = max((off + sizes[i] for i, off in enumerate(offsets)), default=0)

    events = []
    for i in range(n):
        events.append((starts[i], 1, sizes[i]))    # 1 = start
        events.append((ends[i], 0, -sizes[i]))     # 0 = end，同刻先结算 end
    events.sort()
    live = 0
    lower_bound = 0
    k = 0
    while k < len(events):
        t = events[k][0]
        while k < len(events) and events[k][0] == t:
            live += events[k][2]
            k += 1
        lower_bound = max(lower_bound, live)
    return ([int(o) for o in offsets], int(peak), int(lower_bound))


def _assert(got, want):
    assert len(got) == 3, got
    assert [int(x) for x in got[0]] == [int(x) for x in want[0]], (got[0], want[0])
    assert int(got[1]) == int(want[1]), (got[1], want[1])
    assert int(got[2]) == int(want[2]), (got[2], want[2])


def _key_fn(intervals, align):
    def round_up(x):
        return -(-int(x) // align) * align
    return lambda i: (-round_up(intervals[i][2]), int(intervals[i][0]), int(intervals[i][1]), i)


def _rand_intervals(rng, n, span, max_bytes):
    out = []
    for _ in range(n):
        s = rng.randrange(span)
        e = s + rng.randrange(1, span + 1)
        out.append((s, e, rng.randrange(1, max_bytes)))
    return out


# 样例：align=1024，五个张量（size 已对齐）
#   i2 [2,6) 5120 -> 最先放大件：off 0        [0,5120)
#   i0 [0,4) 3072 -> 与 i2 冲突 -> off 5120   [5120,8192)
#   i1 [1,3) 2048 -> 撞 i2、i0  -> off 8192   [8192,10240)
#   i3 [4,6) 2048 -> 只撞 i2，off 5120 可行
#   i4 [5,8) 1024 -> 撞 i2/i3 -> off 7168     [7168,8192)
#   峰值 10240；t=2 时同时存活 i0+i1+i2 = 10240，故下界也是 10240
def check_example(solve):
    iv = [
        (0, 4, 3000),
        (1, 3, 2000),
        (2, 6, 5000),
        (4, 6, 2000),
        (5, 8, 1000),
    ]
    got = solve(iv, 1024)
    _assert(got, ([5120, 8192, 0, 5120, 7168], 10240, 10240))


check_example.example = True


def check_matches_oracle(solve):
    import random

    rng = random.Random(4242)
    for align, span, max_bytes in [(1024, 8, 6000), (1, 6, 9), (256, 10, 2000), (64, 5, 300)]:
        for _ in range(10):
            iv = _rand_intervals(rng, rng.randrange(1, 9), span, max_bytes)
            _assert(solve(iv, align), _oracle(iv, align))


def check_plan_is_valid(solve):
    import random

    rng = random.Random(11)
    for align, span in [(1024, 8), (128, 6), (1, 7)]:
        iv = _rand_intervals(rng, rng.randrange(2, 10), span, 4000)
        offsets, peak, _ = solve(iv, align)
        sizes = [-(-int(x[2]) // align) * align for x in iv]
        for i, off in enumerate(offsets):
            off = int(off)
            assert off % align == 0 and off >= 0, (off, align)
            assert off + sizes[i] <= int(peak), (i, off, sizes[i], peak)
        for i in range(len(iv)):
            for j in range(i + 1, len(iv)):
                if max(int(iv[i][0]), int(iv[j][0])) >= min(int(iv[i][1]), int(iv[j][1])):
                    continue
                a = (int(offsets[i]), int(offsets[i]) + sizes[i])
                b = (int(offsets[j]), int(offsets[j]) + sizes[j])
                assert max(a[0], b[0]) >= min(a[1], b[1]), (i, j, a, b)
        assert int(peak) == max([int(offsets[i]) + sizes[i] for i in range(len(iv))], default=0), peak


def check_peak_not_below_bound(solve):
    import random

    rng = random.Random(2027)
    for align, span in [(1024, 8), (32, 6)]:
        for _ in range(12):
            iv = _rand_intervals(rng, rng.randrange(2, 9), span, 3000)
            _, peak, lb = solve(iv, align)
            assert int(peak) >= int(lb), (iv, peak, lb)


def check_offset_is_minimal(solve):
    # 每个张量放到「再左移一格就必然冲突」的位置：这是算法第 2 步的最小性
    import random

    rng = random.Random(5)
    for align, span in [(1024, 7), (256, 6), (1, 5)]:
        iv = _rand_intervals(rng, rng.randrange(3, 9), span, 3000)
        offsets, _, _ = solve(iv, align)
        key = _key_fn(iv, align)
        order = sorted(range(len(iv)), key=key)
        placed = []
        for i in order:
            size = -(-int(iv[i][2]) // align) * align
            off = int(offsets[i])
            if off > 0:
                blocked = False
                for ps, pe, poff, psize in placed:
                    if max(int(iv[i][0]), ps) >= min(int(iv[i][1]), pe):
                        continue
                    if max(off - align, poff) < min(off - align + size, poff + psize):
                        blocked = True
                        break
                assert blocked, f"下标 {i} 的偏移 {off} 可以再左移一格"
            placed.append((int(iv[i][0]), int(iv[i][1]), off, size))


def check_sequential_reuse(solve):
    iv = [(0, 1, 1024), (1, 2, 1024), (2, 3, 1024)]
    offsets, peak, lb = solve(iv, 256)
    assert [int(x) for x in offsets] == [0, 0, 0], offsets
    assert int(peak) == 1024 and int(lb) == 1024, (peak, lb)


def check_fully_overlapping(solve):
    iv = [(0, 5, 1024), (0, 5, 2048), (0, 5, 512)]
    offsets, peak, lb = solve(iv, 256)
    assert [int(x) for x in offsets] == [2048, 0, 3072], offsets
    assert int(peak) == 3584 and int(lb) == 3584, (peak, lb)


def check_greedy_can_exceed_bound(solve):
    # 贪心不保证最优：i0 [0,1) 与 i1 [2,3) 各占 2 字节且生命周期不重叠，都落在 0；
    # i3 [0,2) 被挤到 2，i2 [1,3) 又被 i1/i3 夹住只能到 3，峰值 4 > 同时存活下界 3
    iv = [(0, 1, 2), (2, 3, 2), (1, 3, 1), (0, 2, 1)]
    offsets, peak, lb = solve(iv, 1)
    assert [int(x) for x in offsets] == [0, 0, 3, 2], offsets
    assert int(peak) == 4 and int(lb) == 3, (peak, lb)


def check_alignment_rounding(solve):
    iv = [(0, 1, 100), (0, 1, 1)]
    offsets, peak, _ = solve(iv, 64)
    assert [int(x) for x in offsets] == [0, 128], offsets
    assert int(peak) == 192, peak
    empty = solve([], 64)
    assert [int(x) for x in empty[0]] == [] and int(empty[1]) == 0 and int(empty[2]) == 0, empty
