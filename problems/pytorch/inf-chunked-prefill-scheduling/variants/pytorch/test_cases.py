def _oracle(prompt_len, max_num_batched_tokens, n_decode_running):
    """独立：逐 token 计数，done 累加，每步取 min(avail, 剩余)。"""
    P = int(prompt_len)
    avail = int(max_num_batched_tokens) - int(n_decode_running)
    res = []
    if avail <= 0:
        return res
    done = 0
    while done < P:
        take = min(avail, P - done)
        res.append(take)
        done += take
    return res


def check_example(solve):
    assert solve(13, 8, 2) == [6, 6, 1], solve(13, 8, 2)


def check_decode_fills_budget(solve):
    # avail<=0 → 一步进不来
    assert solve(10, 3, 5) == []
    assert solve(10, 5, 5) == []


def check_no_decode(solve):
    assert solve(7, 7, 0) == [7]
    assert solve(100, 10, 0) == [10] * 10


def check_zero_prompt(solve):
    assert solve(0, 8, 2) == []


def check_matches_oracle(solve):
    for budget in range(1, 12):
        for ndec in range(0, 12):
            for P in range(0, 40):
                assert solve(P, budget, ndec) == _oracle(P, budget, ndec), (P, budget, ndec)


def check_sum_invariant(solve):
    for P in range(0, 50):
        out = solve(P, 9, 3)
        assert sum(out) == P
        assert all(x > 0 for x in out)


check_example.example = True
