import torch


def _oracle(prompt, observed, n, capacity):
    """同一语义的朴素实现：池用 list 存 (key, table, stamp)，淘汰用 sorted 取头部。"""
    context = [int(x) for x in prompt.tolist()]
    pool = []                                    # [key, {token: count}, stamp]
    clock = 0

    def find(key):
        for entry in pool:
            if entry[0] == key:
                return entry
        return None

    def put(key, token):
        nonlocal clock
        entry = find(key)
        if entry is None:
            entry = [key, {}, clock]
            pool.append(entry)
        entry[1][token] = entry[1].get(token, 0) + 1
        entry[2] = clock
        clock += 1

    def trim():
        nonlocal pool
        dropped = 0
        while len(pool) > capacity:
            order = sorted(pool, key=lambda e: (sum(e[1].values()), e[2], e[0]))
            pool = [entry for entry in pool if entry is not order[0]]
            dropped += 1
        return dropped

    for i in range(len(context) - n):
        put(tuple(context[i:i + n]), context[i + n])
    trim()

    predicted, hits, sizes, evicted = [], [], [], []
    for token in [int(x) for x in observed.tolist()]:
        key = tuple(context[-n:])
        entry = find(key)
        if entry is None:
            predicted.append(-1)
            hits.append(False)
        else:
            ranked = sorted(entry[1].items(), key=lambda kv: (-kv[1], kv[0]))
            predicted.append(ranked[0][0])
            hits.append(True)
        put(key, token)
        context.append(token)
        evicted.append(trim())
        sizes.append(len(pool))
    return predicted, hits, sizes, evicted


def _cmp(got, exp, tag):
    assert got[0].tolist() == list(exp[0]), f"{tag} predicted {got[0].tolist()} vs {exp[0]}"
    assert got[1].tolist() == list(exp[1]), f"{tag} hits {got[1].tolist()} vs {exp[1]}"
    assert got[2].tolist() == list(exp[2]), f"{tag} size {got[2].tolist()} vs {exp[2]}"
    assert got[3].tolist() == list(exp[3]), f"{tag} evicted {got[3].tolist()} vs {exp[3]}"


def check_example(solve):
    torch.manual_seed(0)
    got = solve(torch.tensor([1, 2, 3, 1, 2, 3]), torch.tensor([3, 1, 9, 3, 4]), 2, 4)
    assert got[0].tolist() == [1, -1, 2, -1, -1], f"got {got[0].tolist()}"
    assert got[1].tolist() == [True, False, True, False, False], f"got {got[1].tolist()}"
    assert got[2].tolist() == [3, 4, 4, 4, 4], f"got {got[2].tolist()}"
    assert got[3].tolist() == [0, 0, 0, 1, 1], f"got {got[3].tolist()}"


check_example.example = True


def check_prediction_frequency_and_tie(solve):
    """计数最多的后继胜出；并列时取更小的 token id，强化后翻面。"""
    torch.manual_seed(1)
    got = solve(torch.tensor([7, 2, 5, 7, 2, 9, 7, 2]), torch.tensor([5, 7, 2, 5]), 2, 10)
    assert got[0].tolist() == [5, 7, 2, 5], f"got {got[0].tolist()}"
    assert got[1].tolist() == [True, True, True, True], f"got {got[1].tolist()}"
    assert got[2].tolist() == [5, 5, 5, 5], f"全部命中时 key 数不变，got {got[2].tolist()}"
    assert got[3].tolist() == [0, 0, 0, 0], f"got {got[3].tolist()}"


def check_capacity_keeps_frequent_and_ages_ties(solve):
    """容量 1：高频 key 不会被一次性新 key 挤掉；容量 2：等计数时最旧的先走。"""
    torch.manual_seed(2)
    got = solve(torch.tensor([1, 2, 1, 3]), torch.tensor([2, 2]), 1, 1)
    assert got[0].tolist() == [-1, -1], f"got {got[0].tolist()}"
    assert got[2].tolist() == [1, 1], f"容量 1 时池大小恒为 1，got {got[2].tolist()}"
    assert got[3].tolist() == [1, 1], f"新学的一次性 key 应立刻被淘汰，got {got[3].tolist()}"

    got = solve(torch.tensor([1, 2, 3]), torch.tensor([4, 5]), 1, 2)
    assert got[2].tolist() == [2, 2], f"got {got[2].tolist()}"
    assert got[3].tolist() == [1, 1], f"等计数时每步淘汰最旧的一个，got {got[3].tolist()}"


def check_miss_then_learn(solve):
    """未命中会把当步的 n-gram 学进池，之后同样的上下文就能命中。"""
    torch.manual_seed(3)
    got = solve(torch.tensor([1, 2, 3]), torch.tensor([3, 1, 2, 3, 3]), 2, 8)
    assert got[0].tolist() == [-1, -1, -1, 3, 3], f"got {got[0].tolist()}"
    assert got[1].tolist() == [False, False, False, True, True], f"got {got[1].tolist()}"
    assert got[2].tolist() == [2, 3, 4, 4, 4], f"got {got[2].tolist()}"
    assert got[3].tolist() == [0, 0, 0, 0, 0], f"got {got[3].tolist()}"


def check_matches_oracle_and_errors(solve):
    """随机小规模对拍（含 n>1、容量抖动）+ 非法输入。"""
    torch.manual_seed(21)
    for n, capacity, prompt_len, steps in ((1, 3, 6, 8), (2, 2, 7, 6), (3, 8, 9, 10)):
        prompt = torch.randint(0, 4, (prompt_len,))
        observed = torch.randint(0, 4, (steps,))
        _cmp(solve(prompt, observed, n, capacity), _oracle(prompt, observed, n, capacity),
             f"n={n},cap={capacity}")

    prompt = torch.tensor([1, 2, 3])
    observed = torch.tensor([4])
    bad_calls = [
        (prompt, observed, 0, 4),                       # n < 1
        (prompt, observed, 2, 0),                       # capacity < 1
        (torch.tensor([1, 2]), observed, 2, 4),         # prompt 太短（切不出 n-gram）
        (torch.zeros(2, 2, dtype=torch.int64), observed, 1, 4),   # prompt 非一维
        (prompt, torch.zeros(2, 2, dtype=torch.int64), 1, 4),     # observed 非一维
    ]
    for args in bad_calls:
        try:
            solve(*args)
            raise AssertionError("非法输入应抛 ValueError")
        except ValueError:
            pass
