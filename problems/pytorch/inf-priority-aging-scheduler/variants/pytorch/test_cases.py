"""独立 oracle：按到达排序的待选列表 + 每步 max() 选人（O(n^2)），与参考实现的集合扫描写法不同。"""

import random


def _oracle(arrivals, service, priority, max_batch, aging_interval, priority_cap):
    n = len(arrivals)
    admit = [-1] * n
    finish = [-1] * n
    left = list(service)
    pending = sorted(range(n), key=lambda i: (arrivals[i], i))
    running = []
    t = 0

    while pending or running:
        free = max_batch - len(running)
        while free > 0 and pending:
            ready = [i for i in pending if arrivals[i] <= t]
            if not ready:
                break
            scored = [
                (
                    min(priority_cap, priority[i] + (t - arrivals[i]) // aging_interval),
                    priority[i],
                    -arrivals[i],
                    -i,
                    i,
                )
                for i in ready
            ]
            _, _, _, _, pick = max(scored)
            pending.remove(pick)
            admit[pick] = t
            running.append(pick)
            free -= 1

        if not running:
            t = pending and arrivals[pending[0]] or t
            continue

        still = []
        for i in running:
            left[i] -= 1
            if left[i] == 0:
                finish[i] = t
            else:
                still.append(i)
        running = still
        t += 1

    return admit, finish, max(admit[i] - arrivals[i] for i in range(n))


def check_example(solve):
    """优先级 0 的老请求靠老化在 cap 允许的范围内反超新到的高优请求。"""
    got = solve([0, 0, 1, 2, 3, 4, 5, 6], [1] * 8, [5, 0, 5, 5, 5, 5, 5, 5], 1, 1, 9)
    assert got[0] == [0, 6, 1, 2, 3, 4, 5, 7], f"admit 顺序不对：{got[0]}"
    assert got[1] == [0, 6, 1, 2, 3, 4, 5, 7], f"finish 不对：{got[1]}"
    assert got[2] == 6, f"最长等待 6 步，got {got[2]}"


check_example.example = True


def check_no_headroom(solve):
    """priority_cap 等于最高基础优先级时，老化没有上升空间，低优请求只能排到最后。"""
    got = solve([0, 0, 1, 2, 3, 4, 5, 6], [1] * 8, [5, 0, 5, 5, 5, 5, 5, 5], 1, 1, 5)
    assert got[0][1] == 7, f"cap=5 时低优请求排到第 7 步，got {got[0]}"
    assert got[2] == 7, f"got {got[2]}"


def check_aging_interval(solve):
    """老化每 aging_interval 步涨一档：间隔大时等待更久。"""
    fast = solve([0, 0, 1, 2, 3, 4, 5, 6], [1] * 8, [5, 0, 5, 5, 5, 5, 5, 5], 1, 1, 9)
    slow = solve([0, 0, 1, 2, 3, 4, 5, 6], [1] * 8, [5, 0, 5, 5, 5, 5, 5, 5], 1, 3, 9)
    assert slow[0][1] >= fast[0][1], f"老化更慢不应更早入选：{slow[0]} vs {fast[0]}"


def check_tie_break(solve):
    """有效优先级打平时：基础优先级高的先，再按到达早、下标小。"""
    got = solve([0, 0, 0], [1, 1, 1], [2, 2, 0], 1, 5, 2)
    assert got[0] == [0, 1, 2], f"两个基础优先级 2 的按下标，got {got[0]}"
    assert got[1] == [0, 1, 2], f"got {got[1]}"


def check_batch_slots(solve):
    """max_batch=2：两个名额在第一步一起发放，先完成的先让位。"""
    got = solve([0, 0, 0], [4, 2, 2], [1, 1, 0], 2, 2, 3)
    assert got[0] == [0, 0, 2], f"got {got[0]}"
    assert got[1] == [3, 1, 3], f"短请求先完成、名额第 2 步才被复用，got {got[1]}"
    assert got[2] == 2, f"got {got[2]}"


def check_idle_jump(solve):
    """时钟必须跳到下一个到达步，不能一步步空转。"""
    got = solve([3, 3], [2, 1], [0, 0], 1, 1, 0)
    assert got == ([3, 5], [4, 5], 2), f"got {got}"


def check_random_against_oracle(solve):
    """随机实例与独立 oracle 全等，并核对服务时长与并发上限。"""
    rng = random.Random(4890)
    for _ in range(300):
        n = rng.randint(1, 7)
        arrivals = [rng.randint(0, 8) for _ in range(n)]
        service = [rng.randint(1, 4) for _ in range(n)]
        priority = [rng.randint(0, 5) for _ in range(n)]
        max_batch = rng.randint(1, 3)
        interval = rng.randint(1, 4)
        cap = max(priority) + rng.randint(0, 3)
        args = (arrivals, service, priority, max_batch, interval, cap)

        got = solve(*args)
        want = _oracle(*args)
        assert got == want, f"{args}: {got} vs {want}"

        admit, finish, max_wait = got
        for i in range(n):
            assert admit[i] >= arrivals[i], f"请求 {i} 在到达前就被准入"
            assert finish[i] == admit[i] + service[i] - 1, f"请求 {i} 的服务时长不对"
        assert max_wait == max(admit[i] - arrivals[i] for i in range(n)), "max_wait 与准入步不符"

        horizon = max(finish) + 1
        for t in range(horizon):
            busy = sum(1 for i in range(n) if admit[i] <= t <= finish[i])
            assert busy <= max_batch, f"第 {t} 步并发 {busy} 超过 max_batch={max_batch}"
