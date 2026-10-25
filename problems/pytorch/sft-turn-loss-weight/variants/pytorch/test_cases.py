import torch


def _oracle(role_ids, assistant_id, turn_weights):
    B, L = role_ids.shape
    w = torch.zeros(B, L, dtype=torch.float32)
    for b in range(B):
        turn = -1
        prev_asst = False
        tid = []
        for i in range(L):
            a = int(role_ids[b, i]) == assistant_id
            if a and not prev_asst:
                turn += 1
            tid.append(turn if a else -1)
            prev_asst = a
        cnt = {}
        for t in tid:
            if t >= 0:
                cnt[t] = cnt.get(t, 0) + 1
        for i, t in enumerate(tid):
            if t >= 0:
                base = 1.0 if turn_weights is None else (
                    float(turn_weights[t]) if t < len(turn_weights) else 1.0)
                w[b, i] = base / cnt[t]
    return w


def _runs(role_row, assistant_id):
    L = role_row.shape[0]
    res = []
    i = 0
    while i < L:
        if int(role_row[i]) == assistant_id:
            j = i
            while j < L and int(role_row[j]) == assistant_id:
                j += 1
            res.append((i, j))
            i = j
        else:
            i += 1
    return res


def check_example(solve):
    role = torch.tensor([[2, 2, 0, 2, 2, 2]])
    out = solve(role, 2)
    expect = torch.tensor([[0.5, 0.5, 0.0, 1 / 3, 1 / 3, 1 / 3]])
    assert torch.allclose(out, expect, atol=1e-6), out.tolist()


check_example.example = True


def check_turn_weights(solve):
    role = torch.tensor([[2, 2, 0, 2, 2, 2]])
    out = solve(role, 2, [2.0, 0.5])
    expect = torch.tensor([[1.0, 1.0, 0.0, 0.5 / 3, 0.5 / 3, 0.5 / 3]])
    assert torch.allclose(out, expect, atol=1e-6), out.tolist()


def check_per_turn_sum_equals_base(solve):
    role = torch.tensor([[2, 2, 2, 0, 2, 1, 2, 2]])
    out = solve(role, 2)
    for (a, b) in _runs(role[0], 2):
        s = float(out[0, a:b].sum())
        assert abs(s - 1.0) < 1e-6, (a, b, s)


def check_no_assistant_zero(solve):
    role = torch.tensor([[0, 1, 0, 3]])
    out = solve(role, 2)
    assert torch.allclose(out, torch.zeros(1, 4), atol=1e-6), out.tolist()


def check_matches_oracle(solve):
    for seed in range(6):
        g = torch.Generator().manual_seed(seed)
        role = torch.randint(0, 4, (3, 9), generator=g)
        for tw in (None, [2.0, 0.5], [1.5]):
            y = solve(role, 2, tw)
            assert torch.allclose(y, _oracle(role, 2, tw), atol=1e-6), (seed, tw)
