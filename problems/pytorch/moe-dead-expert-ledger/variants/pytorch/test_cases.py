import torch


def _oracle(load, beta, death_threshold, revive_threshold, patience, revive_patience):
    """纯 Python 状态机：逐专家、逐 step，用标量 float/int 推进，不借用任何张量算子"""
    steps, experts = load.shape
    rows = load.tolist()
    keep = float(beta)
    rise = 1.0 - keep
    death = float(death_threshold)
    revive = float(revive_threshold)

    ema = [0.0] * experts
    is_dead = [False] * experts
    miss = [0] * experts
    back = [0] * experts
    first = [-1] * experts
    changes = [0] * experts
    dead_total = [0] * experts
    ema_rows, dead_rows, streak_rows = [], [], []

    for step in range(steps):
        observed = []
        for e in range(experts):
            ema[e] = keep * ema[e] + rise * rows[step][e]
            observed.append(ema[e])
        ema_rows.append(observed)
        for e in range(experts):
            value = observed[e]
            if is_dead[e]:
                back[e] = back[e] + 1 if value >= revive else 0
                if back[e] >= revive_patience:
                    is_dead[e] = False
                    back[e] = 0
                    miss[e] = 0
                    changes[e] += 1
            else:
                miss[e] = miss[e] + 1 if value < death else 0
                if miss[e] >= patience:
                    is_dead[e] = True
                    miss[e] = 0
                    back[e] = 0
                    changes[e] += 1
                    if first[e] < 0:
                        first[e] = step
            dead_rows.append(1 if is_dead[e] else 0)
            streak_rows.append(back[e] if is_dead[e] else miss[e])
            dead_total[e] += 1 if is_dead[e] else 0
    make = lambda data, dtype: torch.tensor(data, dtype=dtype)  # noqa: E731
    # 内层按专家展平过，这里切回 [steps, E]
    flat_dead = dead_rows
    flat_streak = streak_rows
    dead_matrix = [flat_dead[step * experts:(step + 1) * experts] for step in range(steps)]
    streak_matrix = [flat_streak[step * experts:(step + 1) * experts] for step in range(steps)]
    return (make(ema_rows, torch.float64), make(dead_matrix, torch.bool),
            make(streak_matrix, torch.int64), make(first, torch.int64),
            make(changes, torch.int64), make(dead_total, torch.int64))


def _case(steps=8, experts=5, seed=1, beta=0.5, death=2.0, revive=3.0,
          patience=2, revive_patience=2, span=8):
    rng = torch.Generator().manual_seed(seed)
    load = torch.randint(0, span, (steps, experts), generator=rng, dtype=torch.int64).to(torch.float64)
    return load, beta, death, revive, patience, revive_patience


def _example():
    load = torch.tensor([[4.0, 0.0, 0.0],
                         [4.0, 0.0, 0.0],
                         [4.0, 0.0, 0.0],
                         [4.0, 0.0, 3.0],
                         [4.0, 0.0, 3.0],
                         [4.0, 0.0, 3.0],
                         [4.0, 0.0, 3.0],
                         [4.0, 0.0, 3.0]], dtype=torch.float64)
    return load, 0.5, 1.0, 2.0, 2, 2


def _verify(solve, args):
    tensors = [v for v in args if isinstance(v, torch.Tensor)]
    before = [v.clone() for v in tensors]
    expected = _oracle(*args)
    result = solve(*args)
    assert type(result) is tuple and len(result) == 6
    names = ("ema", "dead", "streak", "first_dead", "state_changes", "dead_steps")
    for name, actual, target in zip(names, result, expected):
        assert isinstance(actual, torch.Tensor) and actual.device.type == "cpu"
        assert actual.dtype == target.dtype and actual.shape == target.shape, (
            name, actual.dtype, actual.shape, target.shape)
        if target.dtype == torch.float64:
            torch.testing.assert_close(actual, target, rtol=2e-11, atol=2e-11)
        else:
            assert torch.equal(actual, target), (name, actual.tolist(), target.tolist())
    for value, snapshot in zip(tensors, before):
        assert torch.equal(value, snapshot)
    return result


def check_example(solve):
    args = _example()
    ema, dead, streak, first_dead, changes, dead_steps = _verify(solve, args)
    # beta=0.5 且负载是整数：EMA 每步都是精确的二进制分数，能手算
    torch.testing.assert_close(ema, torch.tensor(
        [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.5, 0.0, 0.0], [3.75, 0.0, 1.5],
         [3.875, 0.0, 2.25], [3.9375, 0.0, 2.625], [3.96875, 0.0, 2.8125],
         [3.984375, 0.0, 2.90625]], dtype=torch.float64), rtol=0, atol=0)
    assert dead.int().tolist() == [
        [0, 0, 0], [0, 1, 1], [0, 1, 1], [0, 1, 1],
        [0, 1, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0]]
    assert streak.tolist() == [
        [0, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0],
        [0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    assert first_dead.tolist() == [-1, 1, 1]
    # e1 只死过一次再也没复活；e2 死过一次、又在 s5 复活，所以翻转 2 次
    assert changes.tolist() == [0, 1, 2]
    assert dead_steps.tolist() == [0, 7, 4]


check_example.example = True


def check_matches_oracle(solve):
    combos = [
        dict(steps=6, experts=4, seed=2, beta=0.5, death=2.0, revive=3.0, patience=1, revive_patience=1),
        dict(steps=9, experts=3, seed=3, beta=0.9, death=1.0, revive=4.0, patience=2, revive_patience=3),
        dict(steps=7, experts=6, seed=5, beta=0.0, death=3.0, revive=3.0, patience=2, revive_patience=1),
        dict(steps=10, experts=5, seed=7, beta=0.25, death=5.0, revive=6.0, patience=3, revive_patience=2),
        dict(steps=8, experts=7, seed=11, beta=0.7, death=2.0, revive=2.0, patience=4, revive_patience=4),
        dict(steps=5, experts=8, seed=13, beta=0.95, death=4.0, revive=7.0, patience=1, revive_patience=5),
        dict(steps=12, experts=2, seed=17, beta=0.5, death=0.0, revive=0.0, patience=2, revive_patience=2),
        dict(steps=12, experts=4, seed=19, beta=0.8, death=6.0, revive=6.5, patience=3, revive_patience=2, span=16),
    ]
    for combo in combos:
        _verify(solve, _case(**combo))
    load = _case(steps=8, experts=5, seed=23)[0]
    wide = torch.cat([load, load], dim=1)
    _verify(solve, (wide[:, :5], 0.5, 2.0, 3.0, 2, 2))
    _verify(solve, (load.t().contiguous().t(), 0.5, 2.0, 3.0, 2, 2))
    _verify(solve, (load.flip(0).flip(0), 0.5, 2.0, 3.0, 2, 2))


def check_state_machine_invariants(solve):
    for seed in range(1, 7):
        patience = 1 + seed % 3
        revive_patience = 1 + (seed + 1) % 3
        args = _case(steps=10, experts=6, seed=seed + 30, beta=0.3 + 0.1 * seed,
                     death=2.0, revive=2.0 + (seed % 3), patience=patience,
                     revive_patience=revive_patience, span=10)
        load = args[0]
        ema, dead, streak, first_dead, changes, dead_steps = _verify(solve, args)
        steps, experts = load.shape
        for e in range(experts):
            column = dead[:, e].tolist()
            counts = streak[:, e].tolist()
            assert int(dead_steps[e]) == sum(column)
            assert int(first_dead[e]) == (column.index(1) if 1 in column else -1)
            flips = sum(1 for s in range(1, steps) if column[s] != column[s - 1]) + column[0]
            assert int(changes[e]) == flips, (e, int(changes[e]), flips)
            for s in range(steps):
                if column[s]:
                    # 死态里复活计数永远达不到阈值（达到了就翻了）
                    assert counts[s] < revive_patience
                    if s > 0 and not column[s - 1]:
                        assert counts[s] == 0                     # 判死当步计数归零
                    if float(ema[s][e]) < 2.0:
                        assert counts[s] == 0                     # 还低于死亡线却算不出复活进度
                else:
                    assert counts[s] < patience
                    if s > 0 and column[s - 1]:
                        assert counts[s] == 0                     # 复活当步计数归零
                    if float(ema[s][e]) >= 2.0:
                        assert counts[s] == 0                     # 不在滞回带下方就没有 miss
                if s > 0 and column[s] == column[s - 1]:
                    assert counts[s] in (0, counts[s - 1] + 1), (s, counts)
        # ema 的递推与冷启动口径：第 0 步就是 (1-beta)*load[0]
        torch.testing.assert_close(ema[0], (1.0 - args[1]) * load[0], rtol=0, atol=1e-15)
        torch.testing.assert_close(ema[1:], args[1] * ema[:-1] + (1.0 - args[1]) * load[1:],
                                   rtol=0, atol=1e-15)


def check_hysteresis_band(solve):
    # beta=0.5, death=2.0, revive=4.0, patience=1, revive_patience=2
    load = torch.tensor([[8.0], [1.0], [1.0], [1.0], [8.0], [8.0]], dtype=torch.float64)
    ema, dead, streak, first_dead, changes, dead_steps = _verify(solve, (load, 0.5, 2.0, 4.0, 1, 2))
    torch.testing.assert_close(ema, torch.tensor(
        [[4.0], [2.5], [1.75], [1.375], [4.6875], [6.34375]], dtype=torch.float64), rtol=0, atol=0)
    # s1 的 EMA 落在 [2,4) 滞回带里：活着但不累计 miss；s3 落在带里：死了但不累计 revive
    assert dead.int().reshape(-1).tolist() == [0, 0, 1, 1, 1, 0]
    assert streak.reshape(-1).tolist() == [0, 0, 0, 0, 1, 0]
    assert first_dead.tolist() == [2] and changes.tolist() == [2] and dead_steps.tolist() == [3]


def check_threshold_boundaries(solve):
    # 阈值相等 -> 没有滞回带；EMA 恰好等于阈值时：alive 侧不算 miss（严格 <），dead 侧算复活（非严格 >=）
    load = torch.tensor([[4.0], [0.0], [3.0], [0.0]], dtype=torch.float64)
    ema, dead, streak, first_dead, changes, dead_steps = _verify(solve, (load, 0.5, 2.0, 2.0, 1, 1))
    torch.testing.assert_close(ema, torch.tensor([[2.0], [1.0], [2.0], [1.0]], dtype=torch.float64),
                               rtol=0, atol=0)
    assert dead.int().reshape(-1).tolist() == [0, 1, 0, 1]
    assert streak.reshape(-1).tolist() == [0, 0, 0, 0]
    assert first_dead.tolist() == [1] and changes.tolist() == [3] and dead_steps.tolist() == [2]
    # beta = 0：EMA 就是这一步的负载本身，没有任何记忆
    load = torch.tensor([[0.0, 5.0], [3.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    ema, dead, streak, first_dead, changes, dead_steps = _verify(solve, (load, 0.0, 1.0, 2.0, 1, 1))
    torch.testing.assert_close(ema, load, rtol=0, atol=0)
    assert dead.int().tolist() == [[1, 0], [0, 1], [1, 1]]
    assert first_dead.tolist() == [0, 1] and dead_steps.tolist() == [2, 2]


def check_patience_and_cold_start(solve):
    # 耐心比整个历史还长：永远判不了死，但 miss 计数要一路涨上去
    load = torch.tensor([[0.0, 9.0], [0.0, 9.0], [0.0, 9.0]], dtype=torch.float64)
    ema, dead, streak, first_dead, changes, dead_steps = _verify(solve, (load, 0.5, 1.0, 2.0, 5, 2))
    assert bool((~dead).all()) and bool((first_dead == -1).all())
    assert bool((changes == 0).all()) and bool((dead_steps == 0).all())
    assert streak.tolist() == [[1, 0], [2, 0], [3, 0]]
    # 冷启动：EMA 初值 0，第一步就落在死亡线以下，patience=1 时立刻判死
    load = torch.tensor([[4.0], [4.0]], dtype=torch.float64)
    ema, dead, streak, first_dead, changes, dead_steps = _verify(solve, (load, 0.9, 3.0, 4.0, 1, 1))
    torch.testing.assert_close(ema, torch.tensor([[0.4], [0.76]], dtype=torch.float64), rtol=0, atol=1e-15)
    assert dead.int().reshape(-1).tolist() == [1, 1] and first_dead.tolist() == [0]
    assert changes.tolist() == [1] and dead_steps.tolist() == [2]
    # 单步历史、单专家
    _verify(solve, (torch.tensor([[2.0]], dtype=torch.float64), 0.5, 1.0, 3.0, 1, 1))
    # 全零负载：所有专家都在 patience 步后死掉，且再也没有复活进度
    load = torch.zeros(6, 3, dtype=torch.float64)
    ema, dead, streak, first_dead, changes, dead_steps = _verify(solve, (load, 0.5, 1.0, 1.0, 3, 1))
    assert first_dead.tolist() == [2, 2, 2] and changes.tolist() == [1, 1, 1]
    # miss 一路数到 3 才判死，之后负载永远是 0，复活进度攒不起来
    assert dead_steps.tolist() == [4, 4, 4]
    assert streak.tolist() == [[1, 1, 1], [2, 2, 2], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    assert bool((ema == 0.0).all())


def check_invalid(solve):
    load = _case(steps=5, experts=3, seed=3)[0]

    def expect_raises(args, note):
        try:
            solve(*args)
        except ValueError:
            return
        raise AssertionError(f"{note} should raise ValueError")

    expect_raises(([1.0, 2.0], 0.5, 1.0, 2.0, 1, 1), "load is not a tensor")
    expect_raises((load.to(torch.float32), 0.5, 1.0, 2.0, 1, 1), "load float32")
    expect_raises((load.reshape(-1), 0.5, 1.0, 2.0, 1, 1), "load 1-D")
    expect_raises((load.unsqueeze(0), 0.5, 1.0, 2.0, 1, 1), "load 3-D")
    expect_raises((torch.zeros(0, 3, dtype=torch.float64), 0.5, 1.0, 2.0, 1, 1), "no steps")
    expect_raises((torch.zeros(3, 0, dtype=torch.float64), 0.5, 1.0, 2.0, 1, 1), "no experts")
    expect_raises((load - 100.0, 0.5, 1.0, 2.0, 1, 1), "negative load")
    expect_raises((load * float("nan"), 0.5, 1.0, 2.0, 1, 1), "nan load")
    expect_raises((load * float("inf"), 0.5, 1.0, 2.0, 1, 1), "inf load")
    for beta in (1.0, 1.5, -0.1, float("nan"), float("inf"), True, "0.5"):
        expect_raises((load, beta, 1.0, 2.0, 1, 1), f"beta={beta!r}")
    for death in (-1.0, float("nan"), float("inf"), True, "1.0"):
        expect_raises((load, 0.5, death, 2.0, 1, 1), f"death_threshold={death!r}")
    for revive in (-1.0, float("nan"), float("inf"), True, "2.0"):
        expect_raises((load, 0.5, 1.0, revive, 1, 1), f"revive_threshold={revive!r}")
    expect_raises((load, 0.5, 3.0, 2.0, 1, 1), "revive below death")
    for patience in (0, -1, 1.5, True, None):
        expect_raises((load, 0.5, 1.0, 2.0, patience, 1), f"patience={patience!r}")
    for revive_patience in (0, -2, 2.0, False, None):
        expect_raises((load, 0.5, 1.0, 2.0, 1, revive_patience), f"revive_patience={revive_patience!r}")
    # 合法边界：beta=0、阈值相等、耐心超过历史长度、阈值全 0
    solve(load, 0.0, 1.0, 1.0, 1, 1)
    solve(load, 0.5, 2.0, 2.0, 2, 2)
    solve(load, 0.5, 1.0, 2.0, 100, 100)
    solve(load, 0.99, 0.0, 0.0, 1, 1)


def check_maximum_shape(solve):
    _verify(solve, _case(steps=32, experts=16, seed=41, beta=0.9,
                         death=3.0, revive=6.0, patience=4, revive_patience=3, span=16))
    _verify(solve, _case(steps=32, experts=16, seed=43, beta=0.0,
                         death=4.0, revive=4.0, patience=1, revive_patience=1, span=8))
