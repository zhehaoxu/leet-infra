import copy
import random


def _oracle(ids, special, sequence, budget, strategy="longest_first", keep="head"):
    protected = sum(special)
    n0, n1 = sequence.count(0), sequence.count(1)
    available = min(len(ids), budget) - protected
    if available < 0:
        raise ValueError
    feasible = [(a, available - a) for a in range(n0 + 1)
                if 0 <= available - a <= n1]
    if strategy == "only_first":
        feasible = [(a, b) for a, b in feasible if b == n1]
    elif strategy == "only_second":
        feasible = [(a, b) for a, b in feasible if a == n0]
    if not feasible:
        raise ValueError
    quotas = min(feasible, key=lambda p: (abs(p[0] - p[1]), -p[0]))
    selected = {i for i, flag in enumerate(special) if flag}
    for s in (0, 1):
        occurrences = [i for i in range(len(ids)) if sequence[i] == s]
        n = len(occurrences)
        if keep == "head":
            ranked = list(range(n))
        elif keep == "tail":
            ranked = list(reversed(range(n)))
        else:
            ranked = sorted(range(n), key=lambda j: (min(j, n - 1 - j), j))
        selected.update(occurrences[j] for j in ranked[:quotas[s]])
    indices = [i for i in range(len(ids)) if i in selected]
    return dict(input_ids=[ids[i] for i in indices],
                special_mask=[special[i] for i in indices],
                sequence_ids=[sequence[i] for i in indices],
                kept_indices=indices, removed_counts=[n0 - quotas[0], n1 - quotas[1]])


def _verify(solve, args):
    snapshot = copy.deepcopy(args)
    expected = _oracle(*args)
    actual = solve(*args)
    assert type(actual) is dict and actual == expected, (actual, expected)
    assert all(type(value) is list for value in actual.values())
    assert args == snapshot, "input mutation"
    return actual


def _invalid(solve, args):
    snapshot = copy.deepcopy(args)
    try:
        solve(*args)
    except ValueError:
        assert args == snapshot
    else:
        raise AssertionError("expected ValueError")


def _pair(a=5, b=4):
    ids = [101] + list(range(10, 10 + a)) + [102] + list(range(20, 20 + b)) + [102]
    seq = [-1] + [0] * a + [-1] + [1] * b + [-1]
    return ids, [s == -1 for s in seq], seq


def check_example(solve):
    result = _verify(solve, (*_pair(), 8, "longest_first", "middle"))
    assert result["input_ids"] == [101, 10, 11, 14, 102, 20, 23, 102]
    assert result["removed_counts"] == [2, 2]


def check_equal_counts_remove_second(solve):
    out = _verify(solve, (*_pair(3, 3), 8))
    assert out["removed_counts"] == [0, 1]


def check_only_first_and_second(solve):
    for strategy in ("only_first", "only_second"):
        for keep in ("head", "tail", "middle"):
            _verify(solve, (*_pair(5, 5), 10, strategy, keep))


def check_zero_retention_no_negative_zero_slice(solve):
    for keep in ("head", "tail", "middle"):
        _verify(solve, (*_pair(4, 3), 3, "longest_first", keep))
        _verify(solve, (*_pair(0, 3), 3, "only_second", keep))


def check_no_truncation_and_larger_budget(solve):
    for keep in ("head", "tail", "middle"):
        for budget in (10, 200):
            _verify(solve, (*_pair(4, 3), budget, "longest_first", keep))


def check_special_values_are_not_ids(solve):
    _verify(solve, ([102, 102, 102, 102, 102], [True, False, False, False, True],
                    [-1, 0, 0, 1, -1], 3, "longest_first", "tail"))


def check_interleaved_sequences(solve):
    seq = [1, -1, 0, 1, 0, -1, 0, 1, 1, -1, 0]
    for keep in ("head", "tail", "middle"):
        _verify(solve, (list(range(len(seq))), [s == -1 for s in seq], seq, 7,
                        "longest_first", keep))


def check_middle_odd_quota_prefers_prefix(solve):
    out = _verify(solve, (*_pair(7, 0), 6, "only_first", "middle"))
    assert out["input_ids"] == [101, 10, 11, 16, 102, 102]


def check_empty_and_all_protected(solve):
    for strategy in ("only_first", "only_second", "longest_first"):
        _verify(solve, ([], [], [], 0, strategy))
        _verify(solve, ([1, 2], [True, True], [-1, -1], 2, strategy))
    _verify(solve, ([1], [False], [1], 0))


def check_single_sequence_and_unbalanced(solve):
    for a, b in ((0, 12), (12, 0), (1, 15), (15, 1)):
        for keep in ("head", "tail", "middle"):
            _verify(solve, (*_pair(a, b), 8, "longest_first", keep))


def check_all_small_count_budgets(solve):
    for a in range(6):
        for b in range(6):
            for budget in range(3, 4 + a + b):
                _verify(solve, (*_pair(a, b), budget))


def check_maximum_length(solve):
    seq = [-1] * 4 + [0, 1] * 98
    for keep in ("head", "tail", "middle"):
        _verify(solve, (list(range(200)), [s == -1 for s in seq], seq, 71,
                        "longest_first", keep))


def check_seeded_random(solve):
    rng = random.Random(82464)
    for _ in range(90):
        seq = [rng.choice((-1, 0, 1)) for _ in range(rng.randrange(50))]
        ids = [rng.randrange(8) for _ in seq]
        budget = rng.randrange(seq.count(-1), len(seq) + 1)
        keep = rng.choice(("head", "tail", "middle"))
        _verify(solve, (ids, [s == -1 for s in seq], seq, budget, "longest_first", keep))


def check_impossible_budget_or_selected_sequence(solve):
    _invalid(solve, (*_pair(), 2))
    _invalid(solve, (*_pair(1, 5), 6, "only_first"))
    _invalid(solve, (*_pair(5, 1), 6, "only_second"))
    _invalid(solve, (*_pair(3, 0), 5, "only_second"))


def check_invalid_options_before_identity(solve):
    for budget in (-1, True, 3.0, None):
        _invalid(solve, ([], [], [], budget))
    _invalid(solve, ([], [], [], 0, "unknown"))
    _invalid(solve, ([], [], [], 0, "only_first", "unknown"))


check_example.example = True
