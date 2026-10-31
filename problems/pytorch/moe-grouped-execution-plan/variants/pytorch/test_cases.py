import math

import torch


def _oracle(hidden, expert_ids, routing_weights, diag):
    """纯 Python 槽位代数：按 (专家, 槽位) 排序、clamp 后取 diag、哨兵行清零、unsort 后按 k 求和"""
    tokens, dim = hidden.shape
    top_k = expert_ids.shape[1]
    experts = diag.shape[0]
    h = hidden.tolist()
    g = diag.tolist()
    ids = expert_ids.tolist()
    w = routing_weights.tolist()
    flat_ids = [ids[t][j] for t in range(tokens) for j in range(top_k)]
    flat_w = [w[t][j] for t in range(tokens) for j in range(top_k)]
    slots = tokens * top_k
    perm = sorted(range(slots), key=lambda s: (flat_ids[s], s))
    sorted_ids = [flat_ids[s] for s in perm]
    token_rows = [s // top_k for s in perm]
    grouped = []
    for pos in range(slots):
        expert = sorted_ids[pos]
        if expert >= experts:
            grouped.append([0.0] * dim)
            continue
        row = h[token_rows[pos]]
        column = g[expert if expert < experts else experts - 1]
        grouped.append([row[c] * column[c] for c in range(dim)])
    counts = [0] * experts
    for expert in flat_ids:
        if expert < experts:
            counts[expert] += 1
    offsets = []
    running = 0
    for count in counts:
        running += count
        offsets.append(running)
    inv_perm = [0] * slots
    for pos, slot in enumerate(perm):
        inv_perm[slot] = pos
    weighted = [[grouped[pos][c] * flat_w[perm[pos]] for c in range(dim)] for pos in range(slots)]
    out = []
    for t in range(tokens):
        row = []
        for c in range(dim):
            row.append(math.fsum(weighted[inv_perm[t * top_k + j]][c] for j in range(top_k)))
        out.append(row)
    make = lambda data, dtype: torch.tensor(data, dtype=dtype)  # noqa: E731
    return (make(out, torch.float64), make(grouped, torch.float64),
            make(perm, torch.int64), make(inv_perm, torch.int64),
            make(token_rows, torch.int64), make(counts, torch.int64),
            make(offsets, torch.int64))


def _case(t=3, k=2, e=4, d=3, seed=5, sentinels=1):
    rng = torch.Generator().manual_seed(seed)
    hidden = torch.rand((t, d), generator=rng, dtype=torch.float64) * 2 - 1
    diag = torch.rand((e, d), generator=rng, dtype=torch.float64) * 2 - 1
    flat = torch.randint(0, e, (t * k,), generator=rng, dtype=torch.int64).tolist()
    for idx in range(min(sentinels, t * k)):
        flat[idx] = e
    expert_ids = torch.tensor(flat, dtype=torch.int64).view(t, k)
    weights = torch.rand((t, k), generator=rng, dtype=torch.float64) * 1.5
    return hidden, expert_ids, weights, diag


def _verify(solve, args):
    before = {id(v): v.clone() for v in args}
    expected = _oracle(*args)
    result = solve(*args)
    assert type(result) is tuple and len(result) == 7
    for actual, target in zip(result, expected):
        assert isinstance(actual, torch.Tensor) and actual.device.type == "cpu"
        assert actual.dtype == target.dtype and actual.shape == target.shape, (
            actual.dtype, actual.shape, target.shape)
        if target.dtype == torch.int64:
            assert torch.equal(actual, target), actual.tolist()
        else:
            torch.testing.assert_close(actual, target, rtol=2e-11, atol=2e-11)
    for key, snapshot in before.items():
        for value in args:
            if id(value) == key:
                assert torch.equal(value, snapshot)
    return result


def check_example(solve):
    hidden = torch.tensor([[1.0, 2.0], [3.0, -1.0]], dtype=torch.float64)
    expert_ids = torch.tensor([[1, 3], [0, 1]], dtype=torch.int64)
    weights = torch.tensor([[0.5, 0.2], [0.25, 1.0]], dtype=torch.float64)
    diag = torch.tensor([[0.5, -1.0], [2.0, 0.0], [-1.0, 3.0]], dtype=torch.float64)
    out, grouped, perm, inv_perm, token_rows, counts, offsets = _verify(
        solve, (hidden, expert_ids, weights, diag))
    # 槽位 id 为 [1, 3(哨兵), 0, 1]，按 (专家, 槽位) 升序 -> 槽位 2, 0, 3, 1
    assert perm.tolist() == [2, 0, 3, 1]
    assert token_rows.tolist() == [1, 0, 1, 0]
    assert inv_perm.tolist() == [1, 3, 0, 2]
    assert counts.tolist() == [1, 2, 0]
    assert offsets.tolist() == [1, 3, 3]
    # grouped[i] = hidden[token_rows[i]] * diag[expert]，哨兵行必须是 0
    torch.testing.assert_close(grouped, torch.tensor(
        [[1.5, 1.0], [2.0, 0.0], [6.0, 0.0], [0.0, 0.0]], dtype=torch.float64),
        rtol=0, atol=0)
    # token0 = 0.5 * [2,0] + 0.2 * 0 = [1,0]；token1 = 0.25 * [1.5,1] + 1.0 * [6,0]
    torch.testing.assert_close(out, torch.tensor(
        [[1.0, 0.0], [6.375, 0.25]], dtype=torch.float64), rtol=0, atol=0)


check_example.example = True


def check_matches_oracle(solve):
    for seed in range(1, 9):
        sentinels = seed % 4
        args = _case(t=2 + seed % 3, k=1 + seed % 3, e=3 + seed % 3,
                     d=1 + seed % 4, seed=seed, sentinels=sentinels)
        _verify(solve, args)
    # 非连续布局也要能算
    hidden, expert_ids, weights, diag = _case(t=4, k=3, e=4, d=2, seed=23, sentinels=2)
    _verify(solve, (hidden.t().contiguous().t(), expert_ids, weights, diag))
    _verify(solve, (hidden, expert_ids, weights.t().contiguous().t(), diag))


def check_plan_invariants(solve):
    hidden, expert_ids, weights, diag = _case(t=5, k=3, e=4, d=3, seed=17, sentinels=4)
    out, grouped, perm, inv_perm, token_rows, counts, offsets = _verify(
        solve, (hidden, expert_ids, weights, diag))
    slots = expert_ids.numel()
    experts = diag.shape[0]
    assert sorted(perm.tolist()) == list(range(slots))
    assert inv_perm[perm].tolist() == list(range(slots))
    assert token_rows.tolist() == [int(p) // 3 for p in perm.tolist()]
    flat = expert_ids.reshape(-1).tolist()
    sorted_ids = [flat[int(p)] for p in perm.tolist()]
    assert sorted_ids == sorted(sorted_ids), sorted_ids
    valid = slots - sum(1 for v in flat if v == experts)
    assert int(counts.sum()) == valid
    assert int(offsets[-1]) == valid
    assert counts.shape[0] == experts
    # 尾部恰好是哨兵行，且 grouped 的哨兵行整行为 0
    sentinel_rows = sorted_ids[valid:]
    assert all(v == experts for v in sentinel_rows), sentinel_rows
    assert bool((grouped[valid:] == 0.0).all())
    # 每个有效行的 grouped 确实等于 hidden[token] * diag[expert]
    for pos in range(valid):
        token = int(token_rows[pos])
        expert = int(sorted_ids[pos])
        torch.testing.assert_close(grouped[pos], hidden[token] * diag[expert],
                                   rtol=0, atol=1e-15)
    # out 与「直接按 token 累加」一致：这条路径不经过排序
    direct = torch.zeros_like(out)
    for token in range(hidden.shape[0]):
        for j in range(3):
            expert = int(expert_ids[token][j])
            if expert < experts:
                direct[token] += float(weights[token][j]) * hidden[token] * diag[expert]
    torch.testing.assert_close(out, direct, rtol=1e-11, atol=1e-11)


def check_all_sentinels_and_zero_weights(solve):
    hidden, expert_ids, weights, diag = _case(t=3, k=2, e=4, d=3, seed=19, sentinels=6)
    flat = expert_ids.reshape(-1).tolist()
    assert all(v == 4 for v in flat), flat
    out, grouped, perm, inv_perm, token_rows, counts, offsets = _verify(
        solve, (hidden, expert_ids, weights, diag))
    assert bool((out == 0.0).all())
    assert bool((grouped == 0.0).all())
    assert int(counts.sum()) == 0 and int(offsets[-1]) == 0
    # 全哨兵时 id 全相等，稳定排序保留原槽位顺序
    assert perm.tolist() == list(range(6)) and inv_perm.tolist() == list(range(6))
    # 权重为 0：out 为 0，但 grouped 仍然是真实的专家输出（两条路径不能混）
    hidden, expert_ids, weights, diag = _case(t=3, k=2, e=4, d=3, seed=19, sentinels=1)
    zeros = torch.zeros_like(weights)
    out, grouped, *_ = _verify(solve, (hidden, expert_ids, zeros, diag))
    assert bool((out == 0.0).all())
    assert not bool((grouped == 0.0).all())


def check_single_paths(solve):
    # k=1：每个 token 只有一个槽位，排序完全决定于专家编号
    hidden = torch.tensor([[2.0, -1.0], [1.0, 1.0], [0.5, 4.0]], dtype=torch.float64)
    expert_ids = torch.tensor([[1], [0], [1]], dtype=torch.int64)
    weights = torch.ones(3, 1, dtype=torch.float64)
    diag = torch.tensor([[3.0, 0.0], [-1.0, 2.0]], dtype=torch.float64)
    out, grouped, perm, inv_perm, token_rows, counts, offsets = _verify(
        solve, (hidden, expert_ids, weights, diag))
    assert perm.tolist() == [1, 0, 2]
    assert inv_perm.tolist() == [1, 0, 2]
    assert token_rows.tolist() == [1, 0, 2]
    assert counts.tolist() == [1, 2] and offsets.tolist() == [1, 3]
    torch.testing.assert_close(out, torch.tensor(
        [[-2.0, -2.0], [3.0, 0.0], [-0.5, 8.0]], dtype=torch.float64), rtol=0, atol=0)
    # 单 token、单专家：只有一个槽位，计划就是它自己
    _verify(solve, (torch.ones(1, 2, dtype=torch.float64),
                    torch.tensor([[0]], dtype=torch.int64),
                    torch.full((1, 1), 0.25, dtype=torch.float64),
                    torch.full((1, 2), 2.0, dtype=torch.float64)))
    # 所有槽位选同一个专家 -> 并列，perm 必须是原顺序
    args = _case(t=3, k=2, e=4, d=3, seed=29, sentinels=0)
    same = torch.full_like(args[1], 2)
    out, grouped, perm, inv_perm, token_rows, counts, offsets = _verify(
        solve, (args[0], same, args[2], args[3]))
    assert perm.tolist() == list(range(6)) and inv_perm.tolist() == list(range(6))
    assert token_rows.tolist() == [0, 0, 1, 1, 2, 2]
    assert counts.tolist() == [0, 0, 6, 0] and offsets.tolist() == [0, 0, 6, 6]
    assert bool((grouped != 0.0).all())


def check_invalid(solve):
    hidden, expert_ids, weights, diag = _case(t=3, k=2, e=4, d=3, seed=7, sentinels=1)
    bound = int(diag.shape[0])

    def expect_raises(args, note):
        try:
            solve(*args)
        except ValueError:
            return
        raise AssertionError(f"{note} should raise ValueError")

    ids = torch.tensor([[0, bound + 1], [1, 2]], dtype=torch.int64)
    expect_raises((hidden, ids, weights, diag), f"id {bound + 1}")
    ids = torch.tensor([[0, -1], [1, 2]], dtype=torch.int64)
    expect_raises((hidden, ids, weights, diag), "negative id")
    expect_raises((hidden, expert_ids.to(torch.int32), weights, diag), "ids int32")
    expect_raises((hidden, expert_ids.to(torch.float64), weights, diag), "ids float64")
    expect_raises((hidden, expert_ids.reshape(-1), weights.reshape(-1), diag), "ids 1-D")
    # 形状不匹配：权重转置、token 数不等、diag 的 D 不等
    expect_raises((hidden, expert_ids, weights.t(), diag), "weights [2,3]")
    small_ids = torch.zeros(2, 2, dtype=torch.int64)
    expect_raises((hidden, small_ids, torch.ones(2, 2, dtype=torch.float64), diag), "2 tokens vs 3")
    expect_raises((hidden, expert_ids, weights, torch.zeros(4, 2, dtype=torch.float64)), "diag D mismatch")
    expect_raises((hidden, expert_ids, weights, torch.zeros(4, dtype=torch.float64)), "diag 1-D")
    expect_raises((hidden, expert_ids, weights, torch.zeros(0, 3, dtype=torch.float64)), "diag E=0")
    expect_raises((hidden, expert_ids, weights, torch.zeros(4, 3, dtype=torch.float32)), "diag float32")
    expect_raises((hidden.to(torch.float32), expert_ids, weights, diag), "hidden float32")
    expect_raises((hidden.reshape(-1), expert_ids, weights, diag), "hidden 1-D")
    expect_raises((hidden[:2], expert_ids, weights, diag), "hidden 2 tokens")
    expect_raises((torch.zeros(0, 3, dtype=torch.float64), torch.zeros(0, 2, dtype=torch.int64),
                   torch.zeros(0, 2, dtype=torch.float64), diag), "empty batch")
    expect_raises(([1.0, 2.0, 3.0], expert_ids, weights, diag), "hidden is not a tensor")
    # 合法边界：全部哨兵、没有哨兵、k=1
    all_sentinel = torch.full_like(expert_ids, bound)
    solve(hidden, all_sentinel, weights, diag)
    no_sentinel = torch.randint(0, bound, (3, 2), generator=torch.Generator().manual_seed(2),
                               dtype=torch.int64)
    solve(hidden, no_sentinel, weights, diag)
    solve(hidden, expert_ids[:, :1], weights[:, :1], diag)


def check_maximum_shape(solve):
    _verify(solve, _case(t=8, k=8, e=8, d=4, seed=41, sentinels=8))
    _verify(solve, _case(t=8, k=8, e=1, d=4, seed=43, sentinels=0))
