import copy
import random


# Independent oracle: a finite-state machine emits one position at a time.
# It does not construct or concatenate the reference implementation's segments.
def _oracle(ids_a, ids_b, style, cls_id, sep_id, add_special_tokens=True):
    if style not in ("bert", "roberta"):
        raise ValueError("invalid style")
    rows = []
    state, index = "start", 0
    while state != "done":
        if state == "start":
            if add_special_tokens:
                rows.append((cls_id, 0, 1, None))
            state = "a"
        elif state == "a":
            if index < len(ids_a):
                rows.append((ids_a[index], 0, 0, 0))
                index += 1
            else:
                state = "close_a"
        elif state == "close_a":
            if add_special_tokens:
                rows.append((sep_id, 0, 1, None))
            state = "done" if ids_b is None else "bridge"
        elif state == "bridge":
            if style == "roberta" and add_special_tokens:
                rows.append((sep_id, 0, 1, None))
            index, state = 0, "b"
        elif state == "b":
            if index < len(ids_b):
                rows.append((ids_b[index], int(style == "bert"), 0, 1))
                index += 1
            else:
                state = "close_b"
        else:
            if add_special_tokens:
                rows.append((sep_id, int(style == "bert"), 1, None))
            state = "done"
    keys = ("input_ids", "token_type_ids", "special_tokens_mask", "sequence_ids")
    return {key: [row[i] for row in rows] for i, key in enumerate(keys)}


def _verify(solve, args, frozen=None):
    snapshot = copy.deepcopy(args)
    expected = _oracle(*args)
    if frozen is not None:
        assert expected == frozen, (expected, frozen)
    actual = solve(*args)
    assert type(actual) is dict and actual == expected, (args, actual, expected)
    assert all(type(value) is list for value in actual.values())
    for key, values in actual.items():
        assert all(type(v) is int or (key == "sequence_ids" and v is None) for v in values)
    assert args == snapshot, "input mutation"
    return actual


def check_bert_pair(solve):
    _verify(solve, ([11, 12], [21], "bert", 101, 102), {
        "input_ids": [101, 11, 12, 102, 21, 102],
        "token_type_ids": [0, 0, 0, 0, 1, 1],
        "special_tokens_mask": [1, 0, 0, 1, 0, 1],
        "sequence_ids": [None, 0, 0, None, 1, None],
    })


def check_roberta_pair(solve):
    _verify(solve, ([11, 12], [21], "roberta", 0, 2), {
        "input_ids": [0, 11, 12, 2, 2, 21, 2],
        "token_type_ids": [0, 0, 0, 0, 0, 0, 0],
        "special_tokens_mask": [1, 0, 0, 1, 1, 0, 1],
        "sequence_ids": [None, 0, 0, None, None, 1, None],
    })


def check_empty_a_none_b(solve):
    for style in ("bert", "roberta"):
        for a in ([], [3, 4]):
            for flag in (True, False):
                _verify(solve, (a, None, style, 101, 102, flag))
        _verify(solve, ([], [8, 9], style, 101, 102))


def check_empty_b_is_pair(solve):
    for style in ("bert", "roberta"):
        for a in ([], [7, 8]):
            for flag in (True, False):
                _verify(solve, (a, [], style, 1, 2, flag))


def check_disable_insertion(solve):
    for style in ("bert", "roberta"):
        for b in (None, [], [4, 5, 6]):
            out = _verify(solve, ([1, 2, 3], b, style, 1, 2, False))
            assert all(s == 0 for s in out["special_tokens_mask"])


def check_id_collisions(solve):
    for style in ("bert", "roberta"):
        for cls, sep in ((7, 7), (0, 1000000), (1000000, 0)):
            for flag in (True, False):
                _verify(solve, ([cls, sep, cls], [sep, cls], style, cls, sep, flag))


def check_asymmetric_sequences(solve):
    for style in ("bert", "roberta"):
        for a, b in (([8], list(range(17))), (list(range(23)), [9])):
            for flag in (True, False):
                _verify(solve, (a, b, style, 101, 102, flag))


def check_native_empty_pair_fixtures(solve):
    # tokenizers 0.23.3 backend.encode('a', ''): the second sequence exists.
    _verify(solve, ([11], [], "bert", 101, 102), {
        "input_ids": [101, 11, 102, 102],
        "token_type_ids": [0, 0, 0, 1],
        "special_tokens_mask": [1, 0, 1, 1],
        "sequence_ids": [None, 0, None, None],
    })
    _verify(solve, ([11], [], "roberta", 0, 2), {
        "input_ids": [0, 11, 2, 2, 2],
        "token_type_ids": [0, 0, 0, 0, 0],
        "special_tokens_mask": [1, 0, 1, 1, 1],
        "sequence_ids": [None, 0, None, None, None],
    })


def check_small_grid_and_seeded_random(solve):
    for style in ("bert", "roberta"):
        for na in range(4):
            for nb in (None, 0, 1, 2, 3):
                for flag in (True, False):
                    b = None if nb is None else [2] * nb
                    _verify(solve, ([1] * na, b, style, 1, 2, flag))
    rng = random.Random(458)
    for _ in range(150):
        a = [rng.randrange(5) for _ in range(rng.randrange(31))]
        b = None if rng.randrange(4) == 0 else [rng.randrange(5) for _ in range(rng.randrange(31))]
        _verify(solve, (a, b, rng.choice(("bert", "roberta")), rng.randrange(5),
                        rng.randrange(5), rng.choice((True, False))))


def check_maximum_lengths(solve):
    for style in ("bert", "roberta"):
        for flag in (True, False):
            _verify(solve, (list(range(30)), list(range(1000000, 999970, -1)),
                            style, 0, 1000000, flag))


def check_invalid_style_before_identity(solve):
    for style in ("", "BERT", "Roberta", "unknown"):
        for a, b in (([], None), ([], []), ([1], [2])):
            for flag in (False, True):
                args = (a, b, style, 1, 2, flag)
                snapshot = copy.deepcopy(args)
                try:
                    solve(*args)
                except ValueError:
                    assert args == snapshot, "mutation on error"
                else:
                    raise AssertionError("invalid style must raise ValueError")


def check_no_mutation_shared_inputs(solve):
    for style in ("bert", "roberta"):
        for flag in (True, False):
            shared = [102, 101, 102]
            args = (shared, shared, style, 101, 102, flag)
            first = _verify(solve, args)
            second = _verify(solve, args)
            assert first == second


check_bert_pair.example = True
check_roberta_pair.example = True
check_native_empty_pair_fixtures.example = True
