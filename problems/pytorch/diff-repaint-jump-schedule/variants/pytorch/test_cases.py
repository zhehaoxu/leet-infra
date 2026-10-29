import torch


def _oracle(num_train, num_inference, jump_length, jump_n_sample):
    """递归版 oracle：回跳后把剩余行程交给一次新的 descending walk。"""
    n = min(num_train, num_inference)
    budget = {j: jump_n_sample - 1 for j in range(0, n - jump_length, jump_length)}
    out = []

    def walk(t):
        while t >= 1:
            t -= 1
            out.append(t)
            if budget.get(t, 0) > 0:
                budget[t] -= 1
                for k in range(1, jump_length + 1):
                    out.append(t + k)
                walk(t + jump_length)
                return

    walk(n)
    step = num_train // n
    return [x * step for x in out]


def check_example(solve):
    got = solve(100, 10, 2, 2)
    assert got.dtype == torch.int64 and got.dim() == 1 and got.numel() == 26
    assert got.tolist() == [
        90, 80, 70, 60, 70, 80, 70, 60, 50, 40, 50, 60, 50, 40,
        30, 20, 30, 40, 30, 20, 10, 0, 10, 20, 10, 0,
    ], got.tolist()

    got2 = solve(30, 10, 3, 2)
    assert got2.tolist() == [
        27, 24, 21, 18, 21, 24, 27, 24, 21, 18, 15, 12, 9,
        12, 15, 18, 15, 12, 9, 6, 3, 0, 3, 6, 9, 6, 3, 0,
    ], got2.tolist()


check_example.example = True


def check_no_jumps(solve):
    plain = [90, 80, 70, 60, 50, 40, 30, 20, 10, 0]
    assert solve(100, 10, 1, 1).tolist() == plain
    assert solve(100, 10, 2, 1).tolist() == plain
    # jump_length 不小于步数时没有任何回跳点
    assert solve(100, 10, 10, 5).tolist() == plain
    assert solve(100, 10, 25, 3).tolist() == plain


def check_clamp_and_step_ratio(solve):
    # num_inference_steps 超过训练步数时被夹到训练步数
    got = solve(8, 20, 3, 1)
    assert got.tolist() == _oracle(8, 20, 3, 1), got.tolist()
    assert got.tolist() == [7, 6, 5, 4, 3, 2, 1, 0], got.tolist()
    assert solve(8, 20, 10, 2).tolist() == [7, 6, 5, 4, 3, 2, 1, 0]

    # 步长比例取整数除法
    long = solve(1000, 50, 10, 1)
    assert long.numel() == 50
    assert long.tolist() == [20 * k for k in range(49, -1, -1)], long.tolist()

    jumped = solve(1000, 50, 10, 2)
    assert int(jumped.max()) == 980 and bool((jumped % 20 == 0).all())
    assert jumped.numel() > 50


def check_structure(solve):
    cases = [(100, 10, 2, 2), (30, 10, 3, 2), (37, 12, 4, 3), (12, 12, 1, 4), (5, 5, 2, 2)]
    for num_train, num_inference, jump_length, jump_n_sample in cases:
        n = min(num_train, num_inference)
        step = num_train // n
        values = solve(num_train, num_inference, jump_length, jump_n_sample).tolist()
        assert values[0] == (n - 1) * step, (num_train, num_inference, values[:3])
        assert values[-1] == 0, values[-3:]
        assert all(v % step == 0 for v in values)

        diffs = [b - a for a, b in zip(values, values[1:])]
        assert all(abs(d) == step for d in diffs), diffs
        assert all(v <= (n - 1) * step for v in values), max(values)

        runs, cur = [], 0
        for d in diffs + [-step]:
            if d > 0:
                cur += 1
            elif cur:
                runs.append(cur)
                cur = 0
        assert all(r == jump_length for r in runs), runs
        expected_runs = (jump_n_sample - 1) * len(range(0, n - jump_length, jump_length))
        assert len(runs) == expected_runs, (len(runs), expected_runs)


def check_against_oracle(solve):
    grid = [
        (100, 10, 2, 2), (30, 10, 3, 2), (37, 12, 4, 3), (12, 12, 1, 4),
        (5, 5, 2, 2), (64, 8, 3, 4), (100, 25, 5, 2), (9, 4, 1, 2),
        (7, 3, 2, 3), (1000, 20, 6, 5),
    ]
    for args in grid:
        got = solve(*args).tolist()
        want = _oracle(*args)
        assert got == want, (args, got, want)


def check_bad_input(solve):
    bads = [
        lambda: solve(0, 10, 2, 2),
        lambda: solve(-5, 10, 2, 2),
        lambda: solve(100, 0, 2, 2),
        lambda: solve(100, -1, 2, 2),
        lambda: solve(100, 10, 0, 2),
        lambda: solve(100, 10, -2, 2),
        lambda: solve(100, 10, 2, 0),
        lambda: solve(100, 10, 2, -1),
        lambda: solve(100.0, 10, 2, 2),
        lambda: solve(100, 10.0, 2, 2),
        lambda: solve(100, 10, 2.0, 2),
        lambda: solve(100, 10, 2, 2.0),
        lambda: solve(True, 10, 2, 2),
        lambda: solve(100, True, 2, 2),
        lambda: solve("100", 10, 2, 2),
        lambda: solve(None, 10, 2, 2),
    ]
    for fn in bads:
        try:
            fn()
        except ValueError:
            pass
        else:
            raise AssertionError("非法输入应抛 ValueError")
