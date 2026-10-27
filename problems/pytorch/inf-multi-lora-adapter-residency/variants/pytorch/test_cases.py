import random

import torch


def _oracle(requests, adapter_bytes, capacity, pinned):
    """独立 oracle：用「最近使用时刻」时间戳挑最久未用者，不维护 LRU 链表。"""
    sizes = list(adapter_bytes)
    clock = 0
    stamp = {}
    resident = set(pinned)
    for a in pinned:
        stamp[a] = None          # pinned 不参与替换
    hits, evicted = [], []
    loads = 0
    rejected = 0
    peak = sum(sizes[a] for a in pinned)

    for a in requests:
        clock += 1
        if a in resident:
            hits.append(True)
            evicted.append(-1)
            if stamp[a] is not None:
                stamp[a] = clock
            continue
        hits.append(False)
        if len(resident) >= capacity:
            candidates = [x for x in resident if stamp[x] is not None]
            if not candidates:
                rejected += 1
                evicted.append(-1)
                continue
            victim = min(candidates, key=lambda x: (stamp[x], x))
            resident.discard(victim)
            del stamp[victim]
            evicted.append(victim)
        else:
            evicted.append(-1)
        resident.add(a)
        stamp[a] = clock
        loads += 1
        peak = max(peak, sum(sizes[x] for x in resident))

    return hits, evicted, loads, rejected, peak


def check_example(solve):
    torch.manual_seed(0)
    hit, evicted, loads, rejected, peak = solve(
        torch.tensor([1, 1, 2, 2, 1]),
        torch.tensor([100, 200, 300, 400]),
        2,
        torch.tensor([0]),
    )
    assert hit.tolist() == [False, True, False, True, False], f"got {hit.tolist()}"
    assert evicted.tolist() == [-1, -1, 1, -1, 2], f"换出序列应为 [-1,-1,1,-1,2]，got {evicted.tolist()}"
    assert int(loads) == 3, f"动态加载 3 次，got {int(loads)}"
    assert int(rejected) == 0, f"got {int(rejected)}"
    assert int(peak) == 400, f"峰值应为 100+300=400，got {int(peak)}"


check_example.example = True


def check_pinned_is_protected(solve):
    """pinned 永不被换出；池子全被 pin 满时只能拒掉请求。"""
    torch.manual_seed(1)
    hit, evicted, loads, rejected, peak = solve(
        torch.tensor([0, 0, 3]), torch.tensor([5, 6, 7, 8]), 1, torch.tensor([0])
    )
    assert hit.tolist() == [True, True, False], f"前三步：命中、命中、缺页，got {hit.tolist()}"
    assert evicted.tolist() == [-1, -1, -1], f"pinned 不能被换出，got {evicted.tolist()}"
    assert int(loads) == 0 and int(rejected) == 1, f"got loads={int(loads)} rejected={int(rejected)}"
    assert int(peak) == 5, f"只有被 pin 的 adapter 常驻，got {int(peak)}"

    # 两个被 pin，capacity=2，pinned 占满 -> 全部拒绝
    _, _, loads, rejected, _ = solve(
        torch.tensor([2, 3, 2]), torch.tensor([1, 1, 1, 1]), 2, torch.tensor([0, 1])
    )
    assert int(loads) == 0 and int(rejected) == 3, f"got loads={int(loads)} rejected={int(rejected)}"


def check_lru_recency_refresh(solve):
    """命中会刷新最近使用次序，从而把换出压力转给真正更久没用的那个。"""
    torch.manual_seed(2)
    # capacity=2（无 pinned）：1,2 都在池里，再命中 1 之后，缺页 3 应换出 2 而不是 1
    hit, evicted, loads, _, _ = solve(
        torch.tensor([1, 2, 1, 3]),
        torch.tensor([10, 10, 10, 10]),
        2,
        torch.tensor([], dtype=torch.int64),
    )
    assert hit.tolist() == [False, False, True, False], f"got {hit.tolist()}"
    assert evicted.tolist() == [-1, -1, -1, 2], f"应换出最久未用的 2，got {evicted.tolist()}"
    assert int(loads) == 3, f"got {int(loads)}"


def check_random_against_oracle(solve):
    """随机请求流与「时间戳选区」oracle 对拍。"""
    rng = random.Random(4242)
    for _ in range(50):
        n_adapter = rng.randint(1, 5)
        sizes = [rng.randint(1, 9) for _ in range(n_adapter)]
        capacity = rng.randint(1, n_adapter)
        k = rng.randint(0, capacity)
        pin = rng.sample(range(n_adapter), k)
        reqs = [rng.randrange(n_adapter) for _ in range(rng.randint(0, 12))]
        got = solve(
            torch.tensor(reqs, dtype=torch.int64),
            torch.tensor(sizes, dtype=torch.int64),
            capacity,
            torch.tensor(pin, dtype=torch.int64),
        )
        want = _oracle(reqs, sizes, capacity, pin)
        assert got[0].tolist() == want[0], f"hit 不一致：{got[0].tolist()} vs {want[0]}"
        assert got[1].tolist() == want[1], f"evicted 不一致：{got[1].tolist()} vs {want[1]}"
        assert int(got[2]) == want[2], f"loads 不一致：{int(got[2])} vs {want[2]}"
        assert int(got[3]) == want[3], f"rejected 不一致：{int(got[3])} vs {want[3]}"
        assert int(got[4]) == want[4], f"peak 不一致：{int(got[4])} vs {want[4]}"


def check_invalid_inputs(solve):
    s = torch.tensor([1, 2, 3])
    r = torch.tensor([0])
    e = torch.tensor([], dtype=torch.int64)
    for bad in (
        lambda: solve(r, s, 0, e),
        lambda: solve(r, torch.tensor([]), 1, e),
        lambda: solve(r, torch.tensor([0, 1, 2]), 1, e),
        lambda: solve(torch.tensor([[0]]), s, 1, e),
        lambda: solve(r, s, 1, torch.tensor([5])),
        lambda: solve(torch.tensor([3]), s, 2, e),
        lambda: solve(r, s, 3, torch.tensor([0, 0])),
        lambda: solve(r, s, 1, torch.tensor([0, 1])),
    ):
        try:
            bad()
            raise AssertionError("非法输入应抛 ValueError")
        except ValueError:
            pass
