"""独立 oracle：反复扫描选「当前最高优先级」的剩余片段，而不是先 sort 再遍历。"""


def _oracle(segments, budget, pinned_priority=100):
    pinned_total = sum(tok for _n, tok, pri in segments if pri >= pinned_priority)
    if pinned_total > budget:
        raise ValueError("pinned overflow")
    pool = list(range(len(segments)))
    used, kept = 0, []
    while pool:
        pick = pool[0]
        for i in pool[1:]:
            if segments[i][2] > segments[pick][2]:
                pick = i
        pool.remove(pick)
        if used + segments[pick][1] <= budget:
            used += segments[pick][1]
            kept.append(pick)
    return sorted(kept), used


_SEGMENTS = [
    ("system", 40, 100),
    ("history", 120, 10),
    ("tools", 60, 60),
    ("scratchpad", 30, 40),
]


def check_example(solve):
    kept, used = solve(_SEGMENTS, 200)
    assert kept == [0, 2, 3], f"应保留 system/tools/scratchpad：{kept}"
    assert used == 130, f"used 应为 40+60+30=130，got {used}"


check_example.example = True


def check_priority_beats_chronology(solve):
    """最早的历史段优先级最低，反而被丢掉；返回的下标回到时间顺序。"""
    kept, used = solve(_SEGMENTS, 200)
    assert 1 not in kept, "history 优先级最低且装不下，应被压缩掉"
    assert kept == sorted(kept), f"下标应按时间顺序：{kept}"

    # 预算刚好够 history 但仍不够全部：优先保留高优先级的 tools
    kept, _ = solve(_SEGMENTS, 100)
    assert kept == [0, 2], f"应保留 system+tools：{kept}"


def check_pinned(solve):
    # 钉住片段必须保留；未钉住的按优先级装填
    kept, used = solve([("sys", 90, 100), ("hist", 20, 10)], 100)
    assert (kept, used) == ([0], 90), (kept, used)

    # 提高阈值后 tools 不再钉住，于是按优先级装填后仍能多塞 scratchpad
    kept, used = solve(_SEGMENTS, 100, pinned_priority=200)
    assert (kept, used) == ([0, 2], 100), (kept, used)

    # 钉住总量超预算 → ValueError
    for bad in (
        lambda: solve([("sys", 120, 100)], 100),
        lambda: solve([("a", 60, 100), ("b", 60, 90)], 100, pinned_priority=90),
    ):
        try:
            bad()
            raise AssertionError("钉住片段超预算应抛 ValueError")
        except ValueError:
            pass


def check_skip_continue(solve):
    """装不下的大块只跳过、不终止：后面的小块仍要尝试。"""
    segs = [("a", 60, 90), ("b", 120, 80), ("c", 20, 70)]
    kept, used = solve(segs, 100)
    assert (kept, used) == ([0, 2], 80), f"break 式实现会漏掉 c：{(kept, used)}"

    # tokens=0 的片段永远留下
    kept, used = solve([("z", 0, 0), ("y", 500, 50)], 10)
    assert kept == [0] and used == 0, (kept, used)


def check_ties_and_invalid(solve):
    # 同优先级按原始下标先到先得
    assert solve([("a", 60, 50), ("b", 60, 50)], 100) == ([0], 60)
    # 空片段列表
    assert solve([], 0) == ([], 0)

    # 与选择式 oracle 对拍（确定性伪随机数据）
    segs = []
    for i in range(14):
        tokens = 5 + (i * 17) % 40
        priority = 100 if i % 7 == 0 else (i * 13) % 5 * 10
        segs.append((f"s{i}", tokens, priority))
    for budget in (55, 100, 160, 400):
        got = solve(segs, budget)
        want = _oracle(segs, budget)
        assert got == want, f"budget={budget}: {got} != {want}"

    # 预算为 0：没有钉住片段时什么都留不下（钉住片段超预算则抛 ValueError）
    assert solve([("a", 5, 0)], 0) == ([], 0)

    for bad in (
        lambda: solve("segments", 10),
        lambda: solve([("a", 5, 1)], -1),
        lambda: solve([("a", -5, 1)], 10),
        lambda: solve([("a", 5)], 10),
        lambda: solve([("a", 5, 1.5)], 10),
        lambda: solve([("a", 5, 1)], 10, pinned_priority=1.0),
    ):
        try:
            bad()
            raise AssertionError("非法输入应抛 ValueError")
        except ValueError:
            pass
