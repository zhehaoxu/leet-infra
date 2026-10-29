"""Independent oracle: enumerate full paths, then condition on the chosen suffix."""

import copy
from decimal import Decimal, localcontext
import math
import random


def _paths(text, vocab):
    found = []

    def visit(position, ids):
        if position == len(text):
            found.append(tuple(ids))
            return
        for token_id, token in enumerate(vocab):
            if text.startswith(token, position):
                visit(position + len(token), ids + [token_id])

    visit(0, [])
    return found


def _oracle(text, vocab, log_scores, alpha, uniforms):
    if any(not token for token in vocab) or len(set(vocab)) != len(vocab):
        raise ValueError("invalid vocabulary")
    paths = _paths(text, vocab)
    if not paths:
        raise ValueError("no full path")
    energies = [alpha * math.fsum(log_scores[i] for i in path) for path in paths]
    maximum = max(energies)
    with localcontext() as context:
        context.prec = 80
        weights = [Decimal.from_float(energy - maximum).exp() for energy in energies]
        log_partition = maximum + float(sum(weights).ln())
        active = list(zip(paths, weights))
        reversed_ids = []
        used = 0
        while active[0][0]:
            groups = {}
            for path, weight in active:
                groups.setdefault(path[-1], []).append((path[:-1], weight))
            candidates = sorted(groups)
            denominator = sum(weight for _, weight in active)
            selected = candidates[-1]
            cumulative = Decimal(0)
            for token_id in candidates:
                cumulative += sum(weight for _, weight in groups[token_id]) / denominator
                if cumulative > Decimal.from_float(uniforms[used]):
                    selected = token_id
                    break
            used += 1
            reversed_ids.append(selected)
            active = groups[selected]
    ids = list(reversed(reversed_ids))
    log_probability = alpha * math.fsum(log_scores[i] for i in ids) - log_partition
    return (ids, float(log_partition), float(log_probability), used)


def _assert_result(solve, text, vocab, scores, alpha, uniforms):
    before = copy.deepcopy((text, vocab, scores, alpha, uniforms))
    expected = _oracle(text, vocab, scores, alpha, uniforms)
    state = random.getstate()
    result = solve(text, vocab, scores, alpha, uniforms)
    assert random.getstate() == state, "global random state changed"
    assert (text, vocab, scores, alpha, uniforms) == before, "input mutated"
    assert type(result) is tuple and len(result) == 4, "return a four-item tuple"
    ids, log_partition, log_probability, used = result
    assert type(ids) is list and all(type(i) is int for i in ids)
    assert type(log_partition) is float and type(log_probability) is float
    assert type(used) is int
    assert all(0 <= i < len(vocab) for i in ids)
    assert "".join(vocab[i] for i in ids) == text
    assert used == len(ids) == expected[3] and used <= len(uniforms)
    assert ids == expected[0], f"IDs {ids} != {expected[0]}"
    assert math.isfinite(log_partition) and math.isfinite(log_probability)
    assert math.isclose(log_partition, expected[1], rel_tol=1e-11, abs_tol=1e-10), "logZ"
    assert math.isclose(log_probability, expected[2], rel_tol=1e-11, abs_tol=1e-10), "log probability"
    return result


def _raises(solve, text, vocab, scores, alpha, uniforms):
    before = copy.deepcopy((vocab, scores, uniforms))
    try:
        solve(text, vocab, scores, alpha, uniforms)
    except ValueError:
        assert (vocab, scores, uniforms) == before
        return
    raise AssertionError("expected ValueError")


def check_empty_text_and_vocabulary(solve):
    assert _assert_result(solve, "", [], [], 0.0, []) == ([], 0.0, 0.0, 0)
    assert _assert_result(solve, "", ["x"], [-100.0], 4.0, []) == ([], 0.0, 0.0, 0)
    _raises(solve, "x", [], [], 1.0, [0.0])


def check_unsegmentable(solve):
    for text, vocab in (("abc", ["a", "bcx"]), ("abc", ["bc"]), ("ab", ["a"]),
                        ("中a", ["中", "b"]), ("aaa", ["aa"])):
        _raises(solve, text, vocab, [-1.0] * len(vocab), 1.0, [0.5] * len(text))


def check_alpha_zero_global_paths(solve):
    # Three full paths: a+a+a, a+aa, aa+a. Final a has mass 2/3, not 1/2.
    result = _assert_result(solve, "aaa", ["a", "aa"], [-90.0, 0.0], 0.0, [0.6, 0.75, 0.0])
    assert result[0] == [1, 0] and result[3] == 2
    assert math.isclose(result[1], math.log(3.0))


def check_score_changes_posterior(solve):
    # Same uniforms choose different paths as token scores change.
    left = _assert_result(solve, "aa", ["a", "aa"], [-0.1, -3.0], 1.0, [0.5, 0.9])
    right = _assert_result(solve, "aa", ["a", "aa"], [-3.0, -0.1], 1.0, [0.5, 0.9])
    assert left[0] == [0, 0] and right[0] == [1]
    _assert_result(solve, "abab", ["a", "b", "ab", "aba", "bab"],
                   [-0.3, -0.4, -1.5, -0.2, -2.0], 0.25, [0.3, 0.7, 0.2, 0.9])


def check_large_negative_scores(solve):
    result = _assert_result(solve, "a" * 12, ["a"], [-100.0], 4.0, [0.999] * 12)
    assert result == ([0] * 12, -4800.0, 0.0, 12)
    _assert_result(solve, "a" * 12, ["a", "aa"], [-50.0, -100.0], 4.0, [0.37] * 12)


def check_tiny_positive_probability_at_zero(solve):
    result = _assert_result(solve, "ab", ["b", "a", "ab"], [-100.0, -100.0, 0.0],
                            4.0, [0.0, 0.0])
    assert result == ([1, 0], 0.0, -800.0, 2)
    smallest = math.nextafter(0.0, 1.0)
    result = _assert_result(solve, "ab", ["b", "a", "ab"], [-100.0, -100.0, 0.0],
                            4.0, [smallest, 0.0])
    assert result[0] == [2]
    result = _assert_result(solve, "ab", ["b", "a", "ab"], [-90.0, -90.0, 0.0],
                            4.0, [smallest, 0.0])
    assert result[0] == [1, 0]
    result = _assert_result(solve, "a" * 12, ["a", "a" * 12], [-100.0, 0.0],
                            4.0, [0.0] * 12)
    assert result == ([0] * 12, 0.0, -4800.0, 12)


def check_vocabulary_id_order(solve):
    # ID order is deliberately not length or lexicographic order.
    result = _assert_result(solve, "aaa", ["aa", "a", "aaa"], [-1.0] * 3,
                            0.0, [0.2, 0.8, 0.3])
    assert result[0] == [1, 0]
    result = _assert_result(solve, "aaa", ["aaa", "a", "aa"], [-1.0] * 3,
                            0.0, [0.2, 0.8, 0.3])
    assert result[0] == [0]


def check_uniform_extremes(solve):
    result = _assert_result(solve, "aa", ["a", "aa"], [0.0, 0.0], 1.0, [0.0, 0.0])
    assert result[0] == [0, 0]
    near_one = math.nextafter(1.0, 0.0)
    result = _assert_result(solve, "aa", ["a", "aa"], [0.0, 0.0], 1.0, [near_one, near_one])
    assert result[0] == [1]


def check_exact_cdf_boundary(solve):
    for u, ids in ((math.nextafter(0.5, 0.0), [0, 0]),
                   (0.5, [1]), (math.nextafter(0.5, 1.0), [1])):
        result = _assert_result(solve, "aa", ["a", "aa"], [0.0, 0.0], 0.0, [u, 0.0])
        assert result[0] == ids, "CDF equality must advance"


def check_uniform_consumption_and_suffix_conditioning(solve):
    # Final b is forced and still consumes u[0]. u[1] decides the aa prefix.
    result = _assert_result(solve, "aab", ["a", "aa", "b"], [0.0] * 3,
                            0.0, [0.0, 0.75, 0.1])
    assert result[0] == [1, 2] and result[3] == 2
    same = _assert_result(solve, "aab", ["a", "aa", "b"], [0.0] * 3,
                          0.0, [0.0, 0.75, 0.99])
    assert same == result, "unused uniform tail matters"
    result = _assert_result(solve, "abcdef", ["abc", "def"], [-1.0, -2.0], 1.0, [0.1] * 6)
    assert result[0] == [0, 1] and result[3] == 2


def check_unicode_codepoint_matching(solve):
    text = "中\U0001f9e0e\u0301中"
    vocab = ["中", "\U0001f9e0", "e", "\u0301", "中\U0001f9e0", "e\u0301", "é", "\u0301中"]
    _assert_result(solve, text, vocab, [-0.2 * i for i in range(len(vocab))], 1.5, [0.2, 0.8, 0.4, 0.1, 0.9])
    _assert_result(solve, "é", ["é", "e\u0301"], [-1.0, 0.0], 2.0, [0.5])
    _raises(solve, "é", ["e", "\u0301"], [-1.0, -1.0], 1.0, [0.5])


def check_random_full_path_oracle(solve):
    rng = random.Random(462)
    for _ in range(110):
        text = "".join(rng.choice("ab") for _ in range(rng.randrange(1, 9)))
        pool = sorted({text[start:end] for start in range(len(text))
                       for end in range(start + 1, min(len(text), start + 4) + 1)})
        # Keep single code points so every case is segmentable.
        vocab = [token for token in pool if len(token) == 1 or rng.random() < 0.6]
        rng.shuffle(vocab)
        vocab = vocab[:20]
        scores = [-rng.uniform(0.0, 5.0) for _ in vocab]
        uniforms = [rng.random() for _ in text]
        _assert_result(solve, text, vocab, scores, rng.choice([0.0, 0.3, 1.0, 4.0]), uniforms)


def check_reachable_prefix_filter(solve):
    # Ending 'bc' matches, but its preceding 'a' is unreachable.
    result = _assert_result(solve, "abc", ["bc", "abc"], [0.0, -100.0], 4.0, [0.0] * 3)
    assert result == ([1], -400.0, 0.0, 1)
    _assert_result(solve, "abcd", ["bcd", "cd", "ab", "abcd"], [0.0, -1.0, -2.0, -4.0],
                   1.0, [0.4, 0.2, 0.8, 0.6])


def check_maximum_paths_and_vocabulary(solve):
    # Every one of the 2^11 boundary subsets is a path; eight unused entries.
    vocab = ["a" * size for size in range(12, 0, -1)] + ["z" * size for size in range(1, 9)]
    result = _assert_result(solve, "a" * 12, vocab, [-1.0] * 20, 0.0, [0.31] * 12)
    assert math.isclose(result[1], 11 * math.log(2.0))
    _assert_result(solve, "a" * 12, vocab, [-0.01 * i for i in range(20)], 4.0, [0.73] * 12)


def check_input_invariance_and_rng_independence(solve):
    args = ("aaaa", ["a", "aa", "aaa"], [-0.5, -0.3, -1.0], 1.0, [0.2, 0.9, 0.4, 0.7])
    first = _assert_result(solve, *args)
    # A solver must use the supplied stream, never consult random APIs.
    names = ("random", "uniform", "choice", "choices", "randrange", "randint", "getrandbits")
    saved = {name: getattr(random, name) for name in names}

    def forbidden(*args, **kwargs):
        raise AssertionError("global RNG was consulted")

    try:
        for name in names:
            setattr(random, name, forbidden)
        assert _assert_result(solve, *args) == first
    finally:
        for name, function in saved.items():
            setattr(random, name, function)


def check_invalid_vocabulary_even_for_empty_text(solve):
    for text in ("", "a"):
        for vocab in ([""], ["a", ""], ["a", "a"], ["中", "中"]):
            _raises(solve, text, vocab, [-1.0] * len(vocab), 1.0, [0.5] * len(text))


check_alpha_zero_global_paths.example = True
check_exact_cdf_boundary.example = True
check_empty_text_and_vocabulary.example = True
