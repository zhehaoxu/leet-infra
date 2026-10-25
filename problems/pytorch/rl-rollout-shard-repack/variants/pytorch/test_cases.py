import torch


def _oracle(seq_lens, num_ranks, micro_budget):
    """独立重算，处处换结构：
    - 负载用「把该 rank 已分到的长度重新求和」现算（不维护增量计数器）；
    - 初始归属用 ceil 分片边界成员扫描（不用 floor 除法公式）；
    - 装箱用指针 + 内层 while（不用 for + if）；
    - pad 开销用「逐条补差求和 Σ(max−len)」（不用 条数×最长−总数）。"""
    lens = [int(x) for x in seq_lens.reshape(-1).tolist()]
    n = len(lens)
    w = int(num_ranks)
    m = int(micro_budget)
    dest = [0] * n
    micro = [0] * n
    tokens = [0] * w
    saved = [0] * w
    groups = [[] for _ in range(w)]

    remaining = list(range(n))                   # 选择排序逐个挑最小 (长度, 下标)
    order = []
    while remaining:
        best = remaining[0]
        for cand in remaining[1:]:
            if (lens[cand], cand) < (lens[best], best):
                best = cand
        remaining.remove(best)
        order.append(best)

    for i in order:
        best, best_load = 0, sum(lens[q] for q in groups[0])
        for r in range(1, w):
            load = sum(lens[q] for q in groups[r])
            if load < best_load:
                best, best_load = r, load
        groups[best].append(i)
        dest[i] = best
    for r in range(w):
        tokens[r] = sum(lens[q] for q in groups[r])

    mb = 0
    for r in range(w):
        ids = groups[r]
        if not ids:
            continue
        p = 0
        while p < len(ids):                    # 指针推进式装箱
            micro[ids[p]] = mb
            used = lens[ids[p]]
            p += 1
            while p < len(ids) and used + lens[ids[p]] <= m:
                micro[ids[p]] = mb
                used += lens[ids[p]]
                p += 1
            mb += 1
    for r in range(w):
        if groups[r]:
            longest = max(lens[q] for q in groups[r])
            saved[r] = sum(longest - lens[q] for q in groups[r])

    bnd = [-(-r * n // w) for r in range(1, w)]          # ceil(r*N/W) 边界
    moved = []
    for i in range(n):
        init = 0
        for b in bnd:
            if i >= b:
                init += 1
        moved.append(1 if dest[i] != init else 0)

    def t(x):
        return torch.tensor(x, dtype=torch.long)
    return t(dest), t(micro), t(tokens), t(saved), t(moved)


# ---- 样例用例 --------------------------------------------------------------------
def check_example(solve):
    lens = torch.tensor([30, 10, 50, 20, 40, 60, 15, 25])
    out = solve(lens, 3, 70)
    ref = ([1, 0, 0, 2, 2, 1, 1, 0],
           [2, 0, 1, 4, 4, 3, 2, 0],
           [85, 105, 60],
           [65, 75, 20],
           [1, 0, 0, 1, 1, 0, 1, 1])
    for got, exp in zip(out, ref):
        assert torch.equal(got, torch.tensor(exp, dtype=torch.long)), (got.tolist(), exp)
    assert out[0].dtype == torch.long and out[2].numel() == 3


check_example.example = True


# ---- 退化探针 --------------------------------------------------------------------
def check_equal_lengths_round_robin(solve):
    """等长序列 → 贪心自然轮询；rank 内长度全相等 → pad_saved 恰为 0。"""
    lens = torch.full((6,), 20, dtype=torch.long)
    dest, micro, tokens, saved, moved = solve(lens, 3, 45)
    assert torch.equal(dest, torch.tensor([0, 1, 2, 0, 1, 2])), dest
    assert torch.equal(micro, torch.tensor([0, 1, 2, 0, 1, 2])), micro
    assert torch.equal(tokens, torch.tensor([40, 40, 40])), tokens
    assert torch.equal(saved, torch.zeros(3, dtype=torch.long)), saved
    assert torch.equal(moved, torch.tensor([0, 1, 1, 1, 1, 0])), moved


def check_single_rank_no_moves(solve):
    """W=1：全落 rank0、moved 全 0；预算 10 时按升序装箱 3+5=8 一包、9 另起。"""
    lens = torch.tensor([5, 9, 3])
    dest, micro, tokens, saved, moved = solve(lens, 1, 10)
    assert torch.equal(dest, torch.tensor([0, 0, 0])), dest
    assert torch.equal(micro, torch.tensor([0, 1, 0])), micro
    assert torch.equal(tokens, torch.tensor([17])), tokens
    assert torch.equal(saved, torch.tensor([10])), saved      # 3×9 − 17
    assert torch.equal(moved, torch.tensor([0, 0, 0])), moved


def check_single_sequence(solve):
    """N=1：一条序列独占 rank0 的一个微批，pad_saved=0，其余 rank 全空。"""
    dest, micro, tokens, saved, moved = solve(torch.tensor([17]), 4, 8)
    assert torch.equal(dest, torch.tensor([0])), dest
    assert torch.equal(micro, torch.tensor([0])), micro
    assert torch.equal(tokens, torch.tensor([17, 0, 0, 0])), tokens
    assert torch.equal(saved, torch.zeros(4, dtype=torch.long)), saved
    assert torch.equal(moved, torch.tensor([0])), moved


def check_budget_covers_all_one_pack_per_rank(solve):
    """M ≥ 总 token：每个非空 rank 恰 1 个微批，装箱顺序仍与 dest 一致。"""
    lens = torch.tensor([30, 10, 50, 20, 40, 60, 15, 25])
    dest, micro, tokens, saved, moved = solve(lens, 3, 10 ** 6)
    assert torch.equal(micro, torch.tensor([1, 0, 0, 2, 2, 1, 1, 0])), micro
    assert sorted(set(micro.tolist())) == [0, 1, 2], micro
    assert torch.equal(tokens, torch.tensor([85, 105, 60])), tokens
    assert torch.equal(moved, torch.tensor([1, 0, 0, 1, 1, 0, 1, 1])), moved


def check_overlong_sequence_own_pack(solve):
    """单条长度 > 预算：独占一包且该包超预算（不许拆序列）。"""
    lens = torch.tensor([100, 10, 20, 30])
    dest, micro, tokens, saved, moved = solve(lens, 2, 40)
    assert torch.equal(dest, torch.tensor([1, 0, 1, 0])), dest
    assert torch.equal(micro, torch.tensor([2, 0, 1, 0])), micro
    assert torch.equal(tokens, torch.tensor([40, 120])), tokens
    assert torch.equal(saved, torch.tensor([20, 80])), saved
    assert torch.equal(moved, torch.tensor([1, 0, 0, 1])), moved
    # rank1 的两个包：{20} 与 {100}，后者超出预算 40
    assert int(lens[micro == 2].sum()) == 100


def check_more_ranks_than_sequences(solve):
    """W > N：只有前若干 rank 拿到序列，空 rank 的账为 0。"""
    lens = torch.tensor([7, 3, 5])
    dest, micro, tokens, saved, moved = solve(lens, 5, 100)
    assert torch.equal(dest, torch.tensor([2, 0, 1])), dest
    assert torch.equal(tokens, torch.tensor([3, 5, 7, 0, 0])), tokens
    assert torch.equal(saved, torch.zeros(5, dtype=torch.long)), saved
    assert torch.equal(micro, torch.tensor([2, 0, 1])), micro


def check_empty_dataset(solve):
    dest, micro, tokens, saved, moved = solve(torch.zeros(0, dtype=torch.long), 3, 50)
    assert dest.numel() == 0 and micro.numel() == 0 and moved.numel() == 0
    assert torch.equal(tokens, torch.zeros(3, dtype=torch.long)), tokens
    assert torch.equal(saved, torch.zeros(3, dtype=torch.long)), saved
    assert dest.dtype == torch.long


# ---- 规模用例：与独立 oracle 对拍 -------------------------------------------------
def check_matches_oracle_battery(solve):
    battery = [
        ([30, 10, 50, 20, 40, 60, 15, 25], 3, 70),
        ([30, 10, 50, 20, 40, 60, 15, 25], 4, 33),
        ([1, 1, 1, 1, 1, 1, 1, 1, 1], 4, 2),
        ([13, 7, 19, 2, 23, 5, 17, 11, 3, 29], 3, 40),
        ([100, 1, 1, 1, 1, 100], 2, 50),
        ([8], 1, 1),
        ([8], 7, 1000),
        ([64, 128, 32, 96, 256, 48, 160, 80, 320, 16, 224, 112], 4, 300),
        ([5, 5, 5], 3, 5),
        ([2, 4, 6, 8, 10, 12, 14], 6, 9),
    ]
    for lens, w, m in battery:
        args = (torch.tensor(lens, dtype=torch.long), w, m)
        got = solve(*args)
        ref = _oracle(*args)
        assert torch.equal(got[0], ref[0]), (lens, w, m, got[0], ref[0])
        assert torch.equal(got[1], ref[1]), (lens, w, m, got[1], ref[1])
        assert torch.equal(got[2], ref[2]), (lens, w, m, got[2], ref[2])
        assert torch.equal(got[3], ref[3]), (lens, w, m, got[3], ref[3])
        assert torch.equal(got[4], ref[4]), (lens, w, m, got[4], ref[4])


def check_packing_legality(solve):
    """装箱合法性：每包 token 和 ≤ 预算（除非该包只有一条）；包号连续；rank 内长度不降序。"""
    for trial in range(8):
        n = 1 + (7 * trial) % 23
        lens = 1 + (torch.arange(1, n + 1) * (3 + trial) % 31).to(torch.long)
        w = 1 + trial % 5
        m = 5 + 7 * trial
        dest, micro, tokens, saved, moved = solve(lens, w, m)
        assert int(tokens.sum()) == int(lens.sum()), (tokens, lens)
        ids = sorted(set(micro.tolist()))
        assert ids == list(range(len(ids))), micro
        for b in ids:
            sel = lens[micro == b]
            assert int(sel.sum()) <= m or sel.numel() == 1, (b, sel.tolist(), m)
        for r in range(w):
            sel = [int(x) for x in lens[dest == r]]
            assert int(sum(sel)) == int(tokens[r]), (r, sel, tokens)
            assert int(saved[r]) >= 0
            if sel:
                assert int(saved[r]) == len(sel) * max(sel) - int(sum(sel))
        # 编号约定：rank 升序 ⇒ 微批号区间也升序，且每个 rank 的号是一段连续区间
        lo = []
        for r in range(w):
            ids_r = sorted(set(int(b) for b in micro[dest == r]))
            if ids_r:
                assert ids_r == list(range(ids_r[0], ids_r[-1] + 1)), (r, ids_r)
                lo.append((ids_r[0], ids_r[-1]))
        assert lo == sorted(lo), lo
        assert int(moved.sum()) <= int(lens.numel())


def check_rebalancing_reduces_skew(solve):
    """长尾场景：重分片后各 rank 的 token 极差远小于原始连续分块的极差。"""
    lens = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 200, 100, 50, 20, 10])
    dest, _micro, tokens, _saved, moved = solve(lens, 2, 300)
    block0 = int(lens[: 15 // 2].sum())        # 原始连续分块：rank0 拿前 7 条
    block1 = int(lens[15 // 2:].sum())
    assert block0 == 7 and block1 == 383, (block0, block1)
    assert int(tokens.max()) - int(tokens.min()) < block1 - block0, tokens
    assert int(moved.sum()) > 0, moved
