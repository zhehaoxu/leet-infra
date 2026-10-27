import random

import torch


def _oracle(arrival, num_tokens, deadline, max_batch, max_wait, max_steps):
    """独立 oracle：用「最后运行步」的闭式差算 wait，不维护计数器。"""
    n = len(arrival)
    remain = list(num_tokens)
    start = [-1] * n
    finish = [-1] * n
    last = [None] * n
    promotions = 0

    for step in range(max_steps):
        ready = [r for r in range(n) if arrival[r] <= step and finish[r] < 0]
        if not ready:
            continue

        def waited(r):
            base = last[r] if last[r] is not None else arrival[r] - 1
            return step - base - 1

        aged = sorted([r for r in ready if waited(r) >= max_wait], key=lambda r: (deadline[r], r))
        fresh = sorted([r for r in ready if waited(r) < max_wait], key=lambda r: (deadline[r], r))
        chosen = (aged + fresh)[:max_batch]
        promotions += min(len(chosen), len(aged))
        for r in chosen:
            if start[r] < 0:
                start[r] = step
            last[r] = step
            remain[r] -= 1
            if remain[r] == 0:
                finish[r] = step

    missed = sum(1 for r in range(n) if finish[r] < 0 or finish[r] > deadline[r])
    return start, finish, missed, promotions


def check_example(solve):
    torch.manual_seed(0)
    start, finish, missed, promotions = solve(
        torch.tensor([0, 0]),
        torch.tensor([3, 1]),
        torch.tensor([100, 100]),
        1,
        2,
        6,
    )
    assert start.tolist() == [0, 2], f"start 应为 [0, 2]，got {start.tolist()}"
    assert finish.tolist() == [3, 2], f"finish 应为 [3, 2]，got {finish.tolist()}"
    assert int(missed) == 0, f"没超期，got {int(missed)}"
    assert int(promotions) == 1, f"r1 被老化提前一次，got {int(promotions)}"


check_example.example = True


def check_edf_orders_by_deadline(solve):
    """关闭老化后，纯 EDF：deadline 小的先跑，同 deadline 按 id。"""
    torch.manual_seed(1)
    start, finish, missed, promotions = solve(
        torch.tensor([0, 0, 0]),
        torch.tensor([2, 1, 1]),
        torch.tensor([10, 1, 5]),
        1,
        99,
        8,
    )
    assert start.tolist() == [2, 0, 1], f"EDF 顺序应为 r1,r2,r0，got {start.tolist()}"
    assert finish.tolist() == [3, 0, 1], f"got {finish.tolist()}"
    assert int(missed) == 0 and int(promotions) == 0, "全程不超期、不触发老化"

    # 同 deadline 时按 id 升序；max_batch 允许并行
    start, finish, _, _ = solve(
        torch.tensor([0, 0, 0]),
        torch.tensor([1, 1, 1]),
        torch.tensor([5, 5, 5]),
        2,
        99,
        3,
    )
    assert start.tolist() == [0, 0, 1], f"同 deadline 应按 id 取前两个，got {start.tolist()}"
    assert finish.tolist() == [0, 0, 1], f"got {finish.tolist()}"


def check_missed_and_unfinished(solve):
    """deadline 过紧或步数不够都计入 missed，未跑完记 -1。"""
    torch.manual_seed(2)
    start, finish, missed, promotions = solve(
        torch.tensor([0, 0, 0]),
        torch.tensor([2, 2, 2]),
        torch.tensor([0, 1, 2]),
        1,
        5,
        3,
    )
    assert start.tolist() == [0, 2, -1], f"got {start.tolist()}"
    assert finish.tolist() == [1, -1, -1], f"got {finish.tolist()}"
    assert int(missed) == 3, f"三个都没赶在 deadline 前完成，got {int(missed)}"
    assert int(promotions) == 0, f"max_wait=5 不会触发老化，got {int(promotions)}"


def check_random_against_oracle(solve):
    """随机负载与「最后运行步闭式差」的 oracle 对拍。"""
    rng = random.Random(77)
    for _ in range(50):
        n = rng.randint(1, 6)
        arrives = [rng.randint(0, 6) for _ in range(n)]
        tokens = [rng.randint(1, 4) for _ in range(n)]
        dls = [rng.randint(0, 14) for _ in range(n)]
        max_batch = rng.randint(1, 3)
        max_wait = rng.randint(1, 4)
        steps = rng.randint(0, 30)
        got = solve(
            torch.tensor(arrives),
            torch.tensor(tokens),
            torch.tensor(dls),
            max_batch,
            max_wait,
            steps,
        )
        want = _oracle(arrives, tokens, dls, max_batch, max_wait, steps)
        assert got[0].tolist() == want[0], f"start 不一致：{got[0].tolist()} vs {want[0]}"
        assert got[1].tolist() == want[1], f"finish 不一致：{got[1].tolist()} vs {want[1]}"
        assert int(got[2]) == want[2], f"missed 不一致：{int(got[2])} vs {want[2]}"
        assert int(got[3]) == want[3], f"promotions 不一致：{int(got[3])} vs {want[3]}"


def check_invalid_inputs(solve):
    a = torch.tensor([0])
    t = torch.tensor([1])
    d = torch.tensor([1])
    for bad in (
        lambda: solve(a, t, d, 0, 1, 5),
        lambda: solve(a, t, d, 1, 0, 5),
        lambda: solve(a, t, d, 1, 1, -1),
        lambda: solve(torch.tensor([[0]]), t, d, 1, 1, 5),
        lambda: solve(a, torch.tensor([1, 1]), d, 1, 1, 5),
        lambda: solve(torch.tensor([-1]), t, d, 1, 1, 5),
        lambda: solve(a, torch.tensor([0]), d, 1, 1, 5),
        lambda: solve(a, t, torch.tensor([-3]), 1, 1, 5),
    ):
        try:
            bad()
            raise AssertionError("非法输入应抛 ValueError")
        except ValueError:
            pass
