from collections import OrderedDict
import torch


def _oracle(capacity, accesses):
    """独立：OrderedDict 队尾=MRU，驱逐用 popitem(last=False)。final 反转成 MRU→LRU。"""
    cap = int(capacity)
    od = OrderedDict()
    hits = misses = 0
    for a in accesses.tolist():
        a = int(a)
        if cap == 0:
            misses += 1
            continue
        if a in od:
            hits += 1
            od.move_to_end(a)
        else:
            misses += 1
            if len(od) >= cap:
                od.popitem(last=False)
            od[a] = None
            od.move_to_end(a)
    final_mru_first = list(reversed(list(od.keys())))
    return hits, misses, final_mru_first


def check_example(solve):
    h, m, order = solve(3, torch.tensor([1, 2, 3, 4, 1, 2, 5, 1, 2]))
    assert (h, m) == (2, 7), (h, m)
    assert order == [2, 1, 5], order


def check_capacity_zero(solve):
    h, m, order = solve(0, torch.tensor([5, 5, 6]))
    assert (h, m, order) == (0, 3, []), (h, m, order)


def check_capacity_one(solve):
    # cap=1: 只有连续重复才命中
    h, m, order = solve(1, torch.tensor([1, 1, 2, 2, 1]))
    assert (h, m) == (2, 3), (h, m)
    assert order == [1], order


def check_matches_oracle(solve):
    torch.manual_seed(11)
    for _ in range(30):
        cap = int(torch.randint(0, 6, (1,)))
        n = int(torch.randint(1, 20, (1,)))
        accesses = torch.randint(0, 8, (n,))
        got = solve(cap, accesses)
        exp = _oracle(cap, accesses)
        assert got == exp, (cap, accesses.tolist(), got, exp)


check_example.example = True
