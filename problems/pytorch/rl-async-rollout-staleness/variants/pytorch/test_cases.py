import torch

# ---------------------------------------------------------------------------
# 固定测试数据：全部由下标公式确定性生成，判定环境内不产生任何随机数。
# ---------------------------------------------------------------------------

# description.md 的示例（N=8、C=2、step_time=4、B=3、S=2）
EX_LENS = [6, 7, 9, 5, 4, 8, 3, 10]
EX_ARR = [0, 0, 1, 2, 5, 6, 10, 11]

# 24 条：到达时刻无序且有并列（i 与 i+19 同到达，靠请求号打破并列），时长 1..13
LENS_MIX = [(3 * i + 5) % 13 + 1 for i in range(24)]
ARR_MIX = [(5 * i) % 19 for i in range(24)]

# 17 条：到达严格递增（供置换等变性对拍），时长 1..9
LENS_RAMP = [(7 * i + 2) % 9 + 1 for i in range(17)]
ARR_RAMP = [3 * i for i in range(17)]

# 32 条：分组同时刻到达 + 平方公式时长（1..7），并发压满
LENS_BLOCK = [(i * i) % 7 + 1 for i in range(32)]
ARR_BLOCK = [(i % 8) * 4 + i // 8 for i in range(32)]

# 12 条：全部 0 时刻到达（无排队空隙，占用时间线最好核对）
LENS_SYNC = [(2 * i + 3) % 5 + 1 for i in range(12)]
ARR_SYNC = [0] * 12

# 20 条：短 step_time + 小 S 的强截断压力，时长 1..9
LENS_SHORT = [(i * 5) % 9 + 1 for i in range(20)]
ARR_SHORT = [(3 * i + i // 4) % 11 for i in range(20)]

# 40 条：容量 8 的多槽混合，含长尾请求
LENS_BIG = [(11 * i + 4) % 17 + 1 for i in range(40)]
ARR_BIG = [(i * 7) % 23 for i in range(40)]


def _i64(values):
    return torch.tensor(values, dtype=torch.long)


def _select_sort(pairs):
    """朴素选择排序：参考解走 sorted(key=...)，这里逐项比较并弹出，代码路径不同。"""
    rest = list(pairs)
    out = []
    while rest:
        best = 0
        for k in range(1, len(rest)):
            if rest[k] < rest[best]:
                best = k
        out.append(rest.pop(best))
    return out


def _oracle(gen_lens, arrivals, capacity, step_time, batch_size, max_staleness):
    """独立重算：沿时间轴逐格推进的事件循环 + 纯 Python 列表。

    与参考解的结构差异（刻意不同路径，避免同错）：
      * 准入：参考解「一条请求一次算完」，维护每个槽的下一次空闲时刻；本 oracle 让时间
        t = 0,1,2,... 递增，每步先释放 finish<=t 的槽，再从 FCFS 队首尽量多准入。
      * 排队顺序：参考解用 sorted(key=...)；本 oracle 用选择排序 + pop。
      * 名次/成批：参考解对每条请求数名次（双重循环）再按 mask 取组内 max；本 oracle
        反复弹出最小 (finish, 请求号)，再按 B 切块、手工跑最大值。
      * occupancy：参考解逐时刻做张量掩码求和；本 oracle 逐请求在自己的区间上 +1。
    """
    lens = [int(v) for v in list(gen_lens)]
    arr = [int(v) for v in list(arrivals)]
    n = len(lens)
    cap = int(capacity)
    st = int(step_time)
    b = int(batch_size)
    s = int(max_staleness)
    detail = {"start": [], "finish": [], "v0": [], "deadline": [], "truncated": [],
              "batch_id": [], "ready": [], "consume": [], "mu": [],
              "occupancy": [], "stall": [], "num_batches": 0}
    if n == 0:
        return detail

    # 1) FCFS 队列：按 (到达, 请求号) 字典序
    queue = [i for _, i in _select_sort([(arr[i], i) for i in range(n)])]

    # 2) 逐时刻推进：释放 → 准入 → 算截止线与完成
    start = [-1] * n
    finish = [-1] * n
    v0 = [0] * n
    deadline = [0] * n
    trunc = [0] * n
    waiting = list(queue)
    busy = []                                  # 在飞请求各自的完成时刻
    t = 0
    while waiting or busy:
        busy = [f for f in busy if f > t]       # finish 时刻槽即刻可复用
        free = cap - len(busy)
        while free > 0 and waiting and arr[waiting[0]] <= t:
            head = waiting.pop(0)
            start[head] = t
            v0[head] = t // st
            deadline[head] = (v0[head] + s) * st
            full = t + lens[head]
            finish[head] = full if full < deadline[head] else deadline[head]
            trunc[head] = 1 if finish[head] < full else 0
            busy.append(finish[head])
            free -= 1
        if not (waiting or busy):
            break
        if not busy:                            # 全空闲：跳到队首（最早）到达时刻
            t = arr[waiting[0]]
        else:
            t += 1

    # 3) 按 (finish, 请求号) 排队，反复弹出最小者，再顺序切块成批
    seq = [i for _, i in _select_sort([(finish[i], i) for i in range(n)])]
    batch_id = [0] * n
    ready = []
    consume = []
    k = 0
    pos = 0
    while pos < n:
        members = seq[pos:pos + b]
        r = -1
        for m in members:
            if finish[m] > r:
                r = finish[m]
        ready.append(r)
        consume.append(r // st)
        for m in members:
            batch_id[m] = k
        k += 1
        pos += b
    num_batches = k

    # 4) 陈旧度与空转
    mu = [0] * n
    for i in range(n):
        mu[i] = consume[batch_id[i]] - v0[i]
    stall = []
    for j in range(num_batches):
        d = (consume[j] - j) * st
        stall.append(d if d > 0 else 0)

    # 5) 占用时间线：逐请求在自己的 [start, finish) 上自增
    horizon = finish[0]
    for i in range(n):
        if finish[i] > horizon:
            horizon = finish[i]
    occ = [0] * horizon
    for i in range(n):
        for tau in range(start[i], finish[i]):
            occ[tau] += 1

    detail.update({"start": start, "finish": finish, "v0": v0, "deadline": deadline,
                   "truncated": trunc, "batch_id": batch_id, "ready": ready,
                   "consume": consume, "mu": mu, "occupancy": occ, "stall": stall,
                   "num_batches": num_batches})
    return detail


def _oracle_outputs(detail):
    """把 oracle 的中间量表裁剪成题面要求的五元组（int64）。"""
    return (_i64(detail["mu"]), _i64(detail["batch_id"]), _i64(detail["truncated"]),
            _i64(detail["occupancy"]), _i64(detail["stall"]))


def _prep(gen_lens, arrivals, capacity, step_time, batch_size, max_staleness):
    args = (_i64(gen_lens), _i64(arrivals), int(capacity), int(step_time),
            int(batch_size), int(max_staleness))
    detail = _oracle(*args)
    return args, detail, _oracle_outputs(detail)


_NAMES = ("mu", "batch_id", "truncated", "occupancy", "stall")


def _cmp_all(got, exp, ctx):
    assert len(got) == 5, (ctx, len(got))
    for name, g, e in zip(_NAMES, got, exp):
        assert g.dtype == torch.long, (name, "dtype", g.dtype, ctx)
        assert g.dim() == 1, (name, "dim", g.shape, ctx)
        assert torch.equal(g, e), (name, g.tolist(), e.tolist(), ctx)


# ---------------------------------------------------------------------------
# 用例
# ---------------------------------------------------------------------------

def check_example(solve):
    # description.md 的唯一示例：五个输出逐字对照题面（不经 oracle）
    got = solve(_i64(EX_LENS), _i64(EX_ARR), 2, 4, 3, 2)
    exp = (_i64([3, 3, 2, 3, 1, 3, 0, 2]),
           _i64([0, 0, 0, 1, 1, 2, 1, 2]),
           _i64([0, 0, 1, 0, 0, 0, 0, 1]),
           _i64([2] * 20 + [1] * 4),
           _i64([12, 12, 16]))
    _cmp_all(got, exp, "example S=2")
    assert got[3].numel() == 24, got[3].shape        # 长度 = 最晚完成时刻


check_example.example = True


def check_example_staleness_sweep(solve):
    # 题面「同一组数据只改 S」的两行：S=1 与 S=10 的完整输出
    got1 = solve(_i64(EX_LENS), _i64(EX_ARR), 2, 4, 3, 1)
    exp1 = (_i64([2, 2, 1, 2, 1, 1, 1, 1]),
            _i64([0, 0, 0, 1, 1, 1, 2, 2]),
            _i64([1, 1, 1, 1, 0, 1, 0, 1]),
            _i64([2] * 15 + [1]),
            _i64([8, 8, 8]))
    _cmp_all(got1, exp1, "example S=1")

    got10 = solve(_i64(EX_LENS), _i64(EX_ARR), 2, 4, 3, 10)
    exp10 = (_i64([3, 3, 3, 2, 1, 4, 0, 3]),
             _i64([0, 0, 1, 0, 1, 2, 1, 2]),
             _i64([0, 0, 0, 0, 0, 0, 0, 0]),
             _i64([2] * 23 + [1] * 6),
             _i64([12, 12, 20]))
    _cmp_all(got10, exp10, "example S=10")


# 配置表：(gen_lens, arrivals, capacity, step_time, batch_size, max_staleness)
WORKLOADS = [
    (LENS_MIX, ARR_MIX, 3, 5, 4, 2),
    (LENS_MIX, ARR_MIX, 5, 7, 6, 3),
    (LENS_MIX, ARR_MIX, 2, 4, 3, 1),
    (LENS_MIX, ARR_MIX, 8, 13, 5, 32),
    (LENS_BLOCK, ARR_BLOCK, 6, 4, 7, 2),
    (LENS_BLOCK, ARR_BLOCK, 4, 9, 3, 1),
    (LENS_BLOCK, ARR_BLOCK, 11, 3, 10, 4),
    (LENS_BIG, ARR_BIG, 8, 6, 5, 2),
    (LENS_BIG, ARR_BIG, 3, 11, 9, 5),
    (LENS_BIG, ARR_BIG, 16, 2, 4, 1),
    (LENS_SYNC, ARR_SYNC, 4, 3, 5, 2),
    (LENS_SYNC, ARR_SYNC, 1, 5, 3, 32),
    (LENS_SHORT, ARR_SHORT, 5, 2, 3, 1),
    (LENS_SHORT, ARR_SHORT, 2, 3, 7, 2),
    (LENS_RAMP, ARR_RAMP, 3, 4, 4, 2),
    (LENS_RAMP, ARR_RAMP, 7, 8, 1, 3),
]


def check_matches_oracle_workloads(solve):
    # 16 组配置：capacity>1、到达不等且无序、含并列到达，五个输出全等
    for lens, arr, cap, st, bsz, s in WORKLOADS:
        args, _, exp = _prep(lens, arr, cap, st, bsz, s)
        _cmp_all(solve(*args), exp, ("workload", cap, st, bsz, s, len(lens)))


def check_matches_oracle_capacity_sweep(solve):
    # 同一批请求，并发槽 1..9：截断点、完成顺序、成批归属全部跟着变
    for cap in range(1, 10):
        for st, bsz, s in ((4, 5, 2), (2, 3, 1), (9, 4, 6)):
            args, _, exp = _prep(LENS_MIX, ARR_MIX, cap, st, bsz, s)
            _cmp_all(solve(*args), exp, ("cap", cap, st, bsz, s))


def check_matches_oracle_batch_size_sweep(solve):
    # 固定时间线只改 B：批数 = ceil(N/B)，按名次切块必须逐 B 对齐
    n = len(LENS_BIG)
    for bsz in list(range(1, 26)) + [40]:
        args, _, exp = _prep(LENS_BIG, ARR_BIG, 6, 5, bsz, 3)
        got = solve(*args)
        _cmp_all(got, exp, ("B", bsz))
        assert got[4].numel() == (n + bsz - 1) // bsz, (bsz, got[4].numel())
        assert int(got[1].max()) == got[4].numel() - 1, (bsz, got[1].tolist())


def check_matches_oracle_staleness_sweep(solve):
    # S 从 1 扫到 32：截止线放宽 → 截断减少 → 完成顺序改变 → 成批与 mu 全变
    for s in range(1, 33):
        for st, cap, bsz in ((4, 4, 5), (7, 2, 3)):
            args, _, exp = _prep(LENS_SHORT, ARR_SHORT, cap, st, bsz, s)
            _cmp_all(solve(*args), exp, ("S", s, st, cap, bsz))


def check_empty_input(solve):
    # N=0：六个入参给五个空张量输出，全 int64、长度 0
    empty = _i64([])
    got = solve(empty, empty.clone(), 3, 4, 5, 2)
    assert len(got) == 5, len(got)
    for name, g in zip(_NAMES, got):
        assert g.dtype == torch.long, (name, g.dtype)
        assert g.numel() == 0, (name, g.shape)
    _, detail, exp = _prep([], [], 3, 4, 5, 2)
    _cmp_all(got, exp, "N=0")


def check_single_slot_serial(solve):
    # capacity=1：槽永不竞争，occupancy 是只取 0/1 的阶梯函数
    args, detail, exp = _prep(LENS_SYNC, ARR_SYNC, 1, 4, 3, 32)
    got = solve(*args)
    _cmp_all(got, exp, "C=1 同步到达")
    assert torch.equal(got[3], _i64([1] * sum(LENS_SYNC))), got[3].tolist()
    assert got[3].numel() == sum(LENS_SYNC), got[3].shape

    # 到达有空隙时：空槽期 occupancy=0，但峰值恒为 1（串行永不重叠），阶梯只取 0/1
    args2, detail2, exp2 = _prep(LENS_MIX, ARR_MIX, 1, 5, 4, 3)
    got2 = solve(*args2)
    _cmp_all(got2, exp2, "C=1 混合到达")
    assert set(got2[3].tolist()) <= {0, 1}, got2[3].tolist()
    busy_total = sum(detail2["finish"][i] - detail2["start"][i] for i in range(len(LENS_MIX)))
    assert int(got2[3].sum()) == busy_total, (int(got2[3].sum()), busy_total)
    # 单槽时开工时刻 = max(自己到达, 上一条完成)：按 FCFS 名次逐条重算
    order = [i for _, i in _select_sort([(ARR_MIX[i], i) for i in range(24)])]
    prev_free = 0
    for i in order:
        assert detail2["start"][i] == max(ARR_MIX[i], prev_free), (i, detail2["start"][i])
        prev_free = detail2["finish"][i]


def check_loose_staleness_never_truncates(solve):
    # S 取上限且 step_time 很大：截止线远超任何生成时长 ⇒ truncated 全 0、mu 非负
    args, detail, exp = _prep(LENS_MIX, ARR_MIX, 64, 64, 5, 32)
    got = solve(*args)
    _cmp_all(got, exp, "loose S")
    assert torch.equal(got[2], _i64([0] * 24)), got[2].tolist()
    assert (got[0] >= 0).all(), got[0].tolist()
    assert detail["finish"] == [ARR_MIX[i] + LENS_MIX[i] for i in range(24)], detail["finish"]
    assert int(got[3].max()) <= 24 and int(got[3].sum()) == sum(LENS_MIX), got[3].tolist()


def check_capacity_ge_n_starts_at_arrival(solve):
    # 退化不变量：C>=N 且同时到达 ⇒ start 等于到达、occupancy 峰值 = N
    args, detail, exp = _prep(LENS_SYNC, ARR_SYNC, 12, 4, 5, 2)
    got = solve(*args)
    _cmp_all(got, exp, "C>=N")
    assert detail["start"] == [0] * 12, detail["start"]
    assert int(got[3].max()) == 12, got[3].tolist()
    assert got[3].numel() == max(LENS_SYNC), (got[3].numel(), max(LENS_SYNC))
    # 全部 0 时刻开工 ⇒ occupancy[tau] = #{i: gen_len_i > tau}
    assert torch.equal(got[3], _i64([sum(1 for L in LENS_SYNC if L > tau)
                                     for tau in range(max(LENS_SYNC))])), got[3].tolist()


def check_single_batch_when_batch_size_ge_n(solve):
    # B >= N：只有批 0，stall 长度 1，batch_id 全 0，mu 全批共用同一个消费版本
    n = len(LENS_BLOCK)
    st = 6
    for bsz in (n, n + 1, 256):
        args, detail, exp = _prep(LENS_BLOCK, ARR_BLOCK, 7, st, bsz, 3)
        got = solve(*args)
        _cmp_all(got, exp, ("B>=N", bsz))
        assert got[4].numel() == 1, (bsz, got[4].shape)
        assert torch.equal(got[1], _i64([0] * n)), got[1].tolist()
        c0 = detail["consume"][0]
        assert got[4].item() == max(0, c0 * st), (got[4].item(), c0)
        assert torch.equal(got[0], _i64([c0 - detail["v0"][i] for i in range(n)])), got[0].tolist()
        assert int(got[0].min()) >= 0, got[0].tolist()
        assert detail["ready"][0] == max(detail["finish"]), detail["ready"]


def check_staleness_one_bound(solve):
    # S=1 边界：交付不得越过入场版本之后的第一个网格点，且不出现零时长请求
    st = 5
    args, detail, exp = _prep(LENS_MIX, ARR_MIX, 4, st, 3, 1)
    got = solve(*args)
    _cmp_all(got, exp, "S=1")
    for i in range(len(LENS_MIX)):
        assert detail["finish"][i] <= (detail["v0"][i] + 1) * st, (i, detail["finish"][i])
        assert 1 <= detail["finish"][i] - detail["start"][i] <= st, i
        assert detail["deadline"][i] == (detail["v0"][i] + 1) * st, i
    for i, tr in enumerate(detail["truncated"]):
        if tr:
            assert detail["mu"][i] >= 1, (i, detail["mu"][i])
            assert detail["finish"][i] == detail["deadline"][i], i
    # 题面示例的 S=1 行也在这里复验一次
    got_ex = solve(_i64(EX_LENS), _i64(EX_ARR), 2, 4, 3, 1)
    assert torch.equal(got_ex[2], _i64([1, 1, 1, 1, 0, 1, 0, 1])), got_ex[2].tolist()


def check_invariants(solve):
    # 跨配置不变量：mu>=0、0<=occupancy<=C、长度=max finish、截断⇒mu>=S、批号范围、stall 非负
    for lens, arr, cap, st, bsz, s in WORKLOADS:
        args, detail, exp = _prep(lens, arr, cap, st, bsz, s)
        mu, bid, tr, occ, stl = solve(*args)
        n = len(lens)
        ctx = (cap, st, bsz, s)
        assert mu.shape == (n,), (ctx, mu.shape)
        assert bid.shape == (n,) and tr.shape == (n,), (ctx, bid.shape, tr.shape)
        assert stl.shape[0] == (n + bsz - 1) // bsz, (ctx, stl.shape)
        assert (mu >= 0).all(), ("mu nonneg", ctx, mu.tolist())
        assert (stl >= 0).all(), ("stall nonneg", ctx, stl.tolist())
        assert occ.numel() == max(detail["finish"]), (ctx, occ.numel(), max(detail["finish"]))
        assert (occ >= 0).all() and (occ <= cap).all(), ("occ bound", ctx, occ.tolist())
        assert int(occ.sum()) == sum(detail["finish"][i] - detail["start"][i]
                                     for i in range(n)), ("busy mass", ctx)
        assert int(bid.min()) == 0 and int(bid.max()) == stl.numel() - 1, (ctx, bid.tolist())
        assert (tr >= 0).all() and (tr <= 1).all(), ("trunc 0/1", ctx, tr.tolist())
        for i in range(n):
            if tr[i].item() == 1:
                assert mu[i].item() >= s, ("truncated mu", ctx, i, mu.tolist())
            else:
                assert detail["finish"][i] == detail["start"][i] + lens[i], ("cut?", ctx, i)


def check_permutation_equivariance(solve):
    # 到达时刻两两不同时：结果只与请求号有关。倒序输入下，逐条请求的 per-request 输出
    # 按同样的置换对齐；occupancy / stall 是「全局」量，与输入顺序无关。
    cap, st, bsz, s = 4, 5, 3, 2
    n = len(LENS_RAMP)
    args, detail, exp = _prep(LENS_RAMP, ARR_RAMP, cap, st, bsz, s)
    assert len(set(detail["finish"])) == n, ("本用例需要无完成并列", detail["finish"])
    got = solve(*args)
    got_rev = solve(_i64(LENS_RAMP[::-1]), _i64(ARR_RAMP[::-1]), cap, st, bsz, s)
    for j, name in enumerate(_NAMES):
        if name in ("occupancy", "stall"):
            assert torch.equal(got[j], got_rev[j]), (name, got[j].tolist(), got_rev[j].tolist())
        else:
            assert torch.equal(got_rev[j], got[j].flip(0)), (
                name, got_rev[j].tolist(), got[j].flip(0).tolist())
    _cmp_all(got, exp, "ramp 正序")
    # 倒序后的五元组也应等于 oracle 的置换版本
    exp_rev = (_i64(detail["mu"][::-1]), _i64(detail["batch_id"][::-1]),
               _i64(detail["truncated"][::-1]), _i64(detail["occupancy"]),
               _i64(detail["stall"]))
    _cmp_all(got_rev, exp_rev, "ramp 倒序")
    # 另一种置换：交错重排（奇偶分开），同样只按请求号对齐
    idx = [i for i in range(0, n, 2)] + [i for i in range(1, n, 2)]
    perm_args = (_i64([LENS_RAMP[i] for i in idx]), _i64([ARR_RAMP[i] for i in idx]),
                 cap, st, bsz, s)
    got_perm = solve(*perm_args)
    # 新数组第 pos 位放的是原请求 idx[pos]，若逐请求输出按请求号等变，则
    # got_perm[pos] == got[idx[pos]]（用正置换 idx 收集，不是它的逆）。
    for j, name in enumerate(_NAMES):
        if name in ("occupancy", "stall"):
            assert torch.equal(got[j], got_perm[j]), name
        else:
            assert torch.equal(got_perm[j], got[j][_i64(idx)]), (name, j)


def check_demand_surplus_zero_stall(solve):
    # 供给充裕（容量 >= N、时长很短、step_time 很大）：所有批都赶在版本 0 就绪，stall 被 clamp
    lens = [1, 2, 3, 4, 5, 6]
    arr = [0, 0, 0, 0, 0, 0]
    args, detail, exp = _prep(lens, arr, 6, 64, 2, 4)
    got = solve(*args)
    _cmp_all(got, exp, "surplus")
    assert torch.equal(got[4], _i64([0, 0, 0])), got[4].tolist()
    assert torch.equal(got[0], _i64([0] * 6)), got[0].tolist()
    assert detail["consume"] == [0, 0, 0], detail["consume"]
    assert detail["ready"] == [2, 4, 6], detail["ready"]
    assert detail["batch_id"] == [0, 0, 1, 1, 2, 2], detail["batch_id"]


def check_batch_ready_is_short_plank(solve):
    # 短板效应：整批共用批内最晚的 finish，短样本被长样本拖旧
    lens = [4, 4, 20, 1, 1, 1]
    arr = [0, 0, 0, 0, 0, 0]
    st = 4
    args, detail, exp = _prep(lens, arr, 6, st, 2, 32)
    got = solve(*args)
    _cmp_all(got, exp, "short plank")
    n = len(lens)
    for k, rdy in enumerate(detail["ready"]):
        members = [i for i in range(n) if detail["batch_id"][i] == k]
        assert rdy == max(detail["finish"][i] for i in members), (k, rdy, detail["finish"])
        assert detail["consume"][k] == rdy // st, (k, rdy)
        for i in members:
            assert detail["mu"][i] == detail["consume"][k] - detail["v0"][i], i
    # 请求 4（只生成了 1 个单位）与请求 0 同批：二者共享 ready，mu 只差入场版本
    same = [i for i in range(n) if detail["batch_id"][i] == detail["batch_id"][0]]
    assert len(same) > 1, same
    assert min(detail["finish"][i] for i in same) < detail["ready"][detail["batch_id"][0]]
    for i in same:
        assert detail["mu"][i] - detail["mu"][0] == detail["v0"][0] - detail["v0"][i], (
            i, detail["mu"])


def check_partial_rollout_trims_long_tail(solve):
    # 长尾被截止线削平：削到 deadline 的记 truncated=1，没削的不记；削平会改写完成顺序
    lens = [30, 1, 1, 1, 1, 1, 1, 1]
    arr = [0] * 8
    args, detail, exp = _prep(lens, arr, 8, 4, 4, 2)
    got = solve(*args)
    _cmp_all(got, exp, "long tail")
    assert detail["start"][0] == 0 and detail["v0"][0] == 0, detail["start"]
    assert detail["deadline"][0] == 8 and detail["finish"][0] == 8, detail["deadline"]
    assert int(got[2][0]) == 1, got[2].tolist()
    assert int(got[0][0]) >= 2, got[0].tolist()                     # 截断 ⇒ mu >= S
    assert int(got[2][1]) == 0 and detail["finish"][1] == 1, got[2].tolist()
    assert int(got[1][0]) == 1, got[1].tolist()                     # 被削短后跑到了后面

    # 刚好卡在截止线上完成（时长 == 剩余窗口）不算截断
    args2, detail2, exp2 = _prep([8], [0], 1, 4, 1, 2)
    got2 = solve(*args2)
    _cmp_all(got2, exp2, "exact deadline")
    assert int(got2[2][0]) == 0, got2[2].tolist()
    assert detail2["finish"] == [8] and detail2["deadline"] == [8], detail2["finish"]

    # 差一个单位就必被判截断
    args3, detail3, exp3 = _prep([9], [0], 1, 4, 1, 2)
    got3 = solve(*args3)
    _cmp_all(got3, exp3, "one over deadline")
    assert detail3["finish"] == [8] and int(got3[2][0]) == 1, detail3["finish"]


def check_tie_break_by_request_id(solve):
    # 完成时刻全并列：名次靠请求号打破，切块与 ready 都必须稳定
    args, detail, exp = _prep([5, 5, 5, 5], [0, 0, 0, 0], 4, 3, 2, 32)
    got = solve(*args)
    _cmp_all(got, exp, "tie")
    assert detail["finish"] == [5, 5, 5, 5], detail["finish"]
    assert detail["batch_id"] == [0, 0, 1, 1], detail["batch_id"]
    assert detail["ready"] == [5, 5], detail["ready"]
    assert detail["consume"] == [1, 1], detail["consume"]
    assert detail["stall"] == [3, 0], detail["stall"]      # (1-0)*3, clamp((1-1)*3)
    assert torch.equal(got[3], _i64([4] * 5)), got[3].tolist()
    assert torch.equal(got[0], _i64([1, 1, 1, 1])), got[0].tolist()

    # 到达并列同理：(到达, 请求号) 决定谁先占槽
    args2, detail2, exp2 = _prep([3, 9], [4, 4], 1, 4, 1, 32)
    got2 = solve(*args2)
    _cmp_all(got2, exp2, "arrival tie")
    assert detail2["start"] == [4, 7], detail2["start"]     # 请求 0 先占唯一槽
    assert detail2["finish"] == [7, 16], detail2["finish"]

    # 名次并列 + capacity 并列一起出现：同批成员共享 ready，跨批则各算各的
    args3, detail3, exp3 = _prep([2, 2, 8, 8], [0, 0, 0, 0], 4, 4, 2, 32)
    got3 = solve(*args3)
    _cmp_all(got3, exp3, "mixed tie")
    assert detail3["batch_id"] == [0, 0, 1, 1], detail3["batch_id"]
    assert detail3["ready"] == [2, 8], detail3["ready"]
    assert torch.equal(got3[0], _i64([0, 0, 2, 2])), got3[0].tolist()
